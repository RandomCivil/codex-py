from dataclasses import dataclass
from typing import Literal, Protocol

from memory.state import AgentState, ExecutionOutcome, Plan, RecoveryDecision, StepExecution
from agent.executor import Executor

class RecoveryDecisionError(ValueError):
    """The requested recovery decision is not safe or valid."""


class PlannerLike(Protocol):
    async def plan(self, state: AgentState) -> Plan: ...


class ExecutorLike(Protocol):
    async def __aenter__(self) -> "ExecutorLike": ...

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None: ...

    async def execute(self, state: AgentState, revision: int, step_id: str) -> ExecutionOutcome: ...


@dataclass(frozen=True, slots=True)
class AgentResult:
    status: Literal["completed", "blocked"]
    state: AgentState


def mark_stale_execution_interrupted(state: AgentState) -> AgentState:
    """Record uncertain in-progress work before applying a recovery policy."""
    running = next((item for item in state.step_executions if item.status == "running"), None)
    if running is None:
        return state
    return state.with_step_execution(
        StepExecution(
            running.revision,
            running.step_id,
            "interrupted",
            error=running.error or "Step execution was interrupted before recovery",
        )
    )


def apply_recovery_decision(
    state: AgentState, recovery: RecoveryDecision | None
) -> tuple[AgentState, bool]:
    """Apply the only safe disposition of interrupted work.

    The boolean result is true when the caller must return a blocked result.
    """
    state = mark_stale_execution_interrupted(state)
    interrupted = next((item for item in state.step_executions if item.status == "interrupted"), None)
    if interrupted is None:
        return state, False
    if recovery is None:
        raise RecoveryDecisionError("an interrupted Step execution requires --recovery")
    if recovery == "retry":
        raise RecoveryDecisionError("retry is unavailable for non-idempotent MCP tools")
    if recovery == "abort":
        return state, True
    if recovery != "fail":
        raise RecoveryDecisionError(f"invalid recovery decision: {recovery}")
    return (
        state.with_step_execution(
            StepExecution(
                interrupted.revision,
                interrupted.step_id,
                "failed",
                error=interrupted.error or "Step execution was interrupted",
            )
        ),
        False,
    )


class Agent:
    """Coordinate one serial Plan–Execute run for an Agent state."""

    def __init__(self, planner: PlannerLike, executor: ExecutorLike|Executor) -> None:
        self._planner = planner
        self._executor = executor

    async def run(self, state: AgentState, recovery: RecoveryDecision | None = None) -> AgentResult:
        state, abort = apply_recovery_decision(state, recovery)
        if abort:
            return AgentResult("blocked", state)
        async with self._executor as executor:
            while True:
                if not state.plan_history:
                    state = state.with_plan(await self._planner.plan(state))

                plan = state.plan_history[-1]
                failed = any(
                    (execution := _execution_for(state, plan.revision, step.id)) is not None
                    and execution.status == "failed"
                    for step in plan.steps
                )
                if failed:
                    result = await self._replan_after_failure(state, plan)
                    if isinstance(result, AgentResult):
                        return result
                    state = result
                    continue

                for step in plan.steps:
                    # state is a communitation between plan steps
                    execution = _execution_for(state, plan.revision, step.id)
                    if execution is not None:
                        if execution.status == "completed":
                            continue
                        if execution.status == "failed":
                            failed = True
                            break
                    outcome = await executor.execute(state, plan.revision, step.id)
                    execution = outcome.execution
                    state = state.with_step_execution(execution)
                    if execution.status == "failed":
                        failed = True
                        break

                if failed:
                    result = await self._replan_after_failure(state, plan)
                    if isinstance(result, AgentResult):
                        return result
                    state = result
                    continue

                if all(
                    (execution := _execution_for(state, plan.revision, step.id)) is not None
                    and execution.status == "completed"
                    for step in plan.steps
                ):
                    return AgentResult("completed", state)

    async def _replan_after_failure(self, state: AgentState, plan: Plan) -> AgentState | AgentResult:
        if plan.revision == 3:
            return AgentResult("blocked", state)
        return state.with_plan(await self._planner.plan(state))


def _execution_for(state: AgentState, revision: int, step_id: str) -> StepExecution | None:
    return next(
        (
            execution
            for execution in state.step_executions
            if (execution.revision, execution.step_id) == (revision, step_id)
        ),
        None,
    )
