from dataclasses import dataclass
from typing import Literal, Protocol

from memory.state import AgentState, Plan, StepExecution


class PlannerLike(Protocol):
    async def plan(self, state: AgentState) -> Plan: ...


class ExecutorLike(Protocol):
    async def __aenter__(self) -> "ExecutorLike": ...

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None: ...

    async def execute(self, state: AgentState, revision: int, step_id: str) -> StepExecution: ...


@dataclass(frozen=True, slots=True)
class AgentResult:
    status: Literal["completed", "blocked"]
    state: AgentState


class Agent:
    """Coordinate one serial Plan–Execute run for an Agent state."""

    def __init__(self, planner: PlannerLike, executor: ExecutorLike) -> None:
        self._planner = planner
        self._executor = executor

    async def run(self, state: AgentState) -> AgentResult:
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
                    execution = _execution_for(state, plan.revision, step.id)
                    if execution is not None:
                        if execution.status == "completed":
                            continue
                        if execution.status == "failed":
                            failed = True
                            break
                    execution = await executor.execute(state, plan.revision, step.id)
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
