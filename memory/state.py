from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal


ExecutionStatus = Literal["pending", "running", "interrupted", "completed", "failed", "skipped"]
RecoveryDecision = Literal["fail", "abort"]
_EXECUTION_STATUSES = frozenset({"pending", "running", "interrupted", "completed", "failed", "skipped"})


def _require_text(value: str, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _require_revision(value: int, field: str) -> None:
    if type(value) is not int or not 1 <= value <= 3:
        raise ValueError(f"{field} must be between 1 and 3")


@dataclass(frozen=True, slots=True)
class PlanStep:
    id: str
    intent: str
    completion_criterion: str

    def __post_init__(self) -> None:
        _require_text(self.id, "step id")
        _require_text(self.intent, "step intent")
        _require_text(self.completion_criterion, "completion criterion")


@dataclass(frozen=True, slots=True)
class Plan:
    revision: int
    goal: str
    steps: tuple[PlanStep, ...]

    def __post_init__(self) -> None:
        _require_revision(self.revision, "plan revision")
        _require_text(self.goal, "plan goal")
        steps = tuple(self.steps)
        if not steps:
            raise ValueError("plan must contain at least one step")
        if any(not isinstance(step, PlanStep) for step in steps):
            raise TypeError("plan steps must be PlanStep values")
        if len({step.id for step in steps}) != len(steps):
            raise ValueError("plan step IDs must be unique")
        object.__setattr__(self, "steps", steps)


@dataclass(frozen=True, slots=True)
class StepExecution:
    """The durable status and next-step handoff for one Plan step.

    For completed executions, ``result`` is a concise evidence summary derived
    from the executor's graph messages, rather than the raw graph state.
    """

    revision: int
    step_id: str
    status: ExecutionStatus
    result: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        _require_revision(self.revision, "step execution revision")
        _require_text(self.step_id, "step execution step ID")
        if self.status not in _EXECUTION_STATUSES:
            raise ValueError("step execution status is invalid")
        if self.result is not None and not isinstance(self.result, str):
            raise TypeError("step execution result must be a string or None")
        if self.error is not None and not isinstance(self.error, str):
            raise TypeError("step execution error must be a string or None")


def _context_paths(values: tuple[str, ...] | list[str], field: str) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError(f"{field} must be a list or tuple")
    values = tuple(values)
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must contain non-empty strings")
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or value != path.as_posix()
            or value in {".", ".."}
            or "\x00" in value
            or "\\" in value
            or (len(value) >= 2 and value[1] == ":")
            or any(part == ".." for part in path.parts)
        ):
            raise ValueError(f"{field} contains an unsafe path: {value!r}")
    return values


def _context_observations(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("observations must be a list or tuple")
    values = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError("observations must contain non-empty strings")
    return values


@dataclass(frozen=True, slots=True)
class StepContext:
    """The cumulative handoff supplied to an Executor invocation."""

    files_read: tuple[str, ...] = ()
    files_modified: tuple[str, ...] = ()
    observations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "files_read", _context_paths(self.files_read, "files_read"))
        object.__setattr__(self, "files_modified", _context_paths(self.files_modified, "files_modified"))
        object.__setattr__(self, "observations", _context_observations(self.observations))


@dataclass(frozen=True, slots=True)
class ContextUpdate(StepContext):
    """The validated context reported by one successful Step execution."""


@dataclass(frozen=True, slots=True)
class ExecutionOutcome:
    """The Executor result and the successful step's optional context update."""

    execution: StepExecution
    context_update: ContextUpdate | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.execution, StepExecution):
            raise TypeError("execution must be a StepExecution")
        if self.context_update is not None and not isinstance(self.context_update, ContextUpdate):
            raise TypeError("context_update must be a ContextUpdate or None")
        if self.execution.status == "completed" and self.context_update is None:
            raise ValueError("a completed execution requires a ContextUpdate")
        if self.execution.status != "completed" and self.context_update is not None:
            raise ValueError("only a completed execution may contain a ContextUpdate")

@dataclass(frozen=True, slots=True)
class AgentState:
    goal: str
    plan_history: tuple[Plan, ...] = ()
    step_executions: tuple[StepExecution, ...] = ()
    memory_summary: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.goal, "agent state goal")
        plans = tuple(self.plan_history)
        executions = tuple(self.step_executions)
        if any(not isinstance(plan, Plan) for plan in plans):
            raise TypeError("plan history must contain Plan values")
        if [plan.revision for plan in plans] != list(range(1, len(plans) + 1)):
            raise ValueError("plan revisions must be sequential")
        if any(plan.goal != self.goal for plan in plans):
            raise ValueError("plan goal must match agent state goal")
        if len({(execution.revision, execution.step_id) for execution in executions}) != len(executions):
            raise ValueError("step executions must be uniquely associated")
        plan_steps = {(plan.revision, step.id) for plan in plans for step in plan.steps}
        if any((execution.revision, execution.step_id) not in plan_steps for execution in executions):
            raise ValueError("step execution must be associated with a plan step")
        if self.memory_summary is not None and not isinstance(self.memory_summary, str):
            raise TypeError("memory summary must be a string or None")
        object.__setattr__(self, "plan_history", plans)
        object.__setattr__(self, "step_executions", executions)

    def with_plan(self, plan: Plan) -> "AgentState":
        return AgentState(
            goal=self.goal,
            plan_history=self.plan_history + (plan,),
            step_executions=self.step_executions,
            memory_summary=self.memory_summary,
        )

    def with_step_execution(self, execution: StepExecution) -> "AgentState":
        remaining = tuple(
            item
            for item in self.step_executions
            if (item.revision, item.step_id) != (execution.revision, execution.step_id)
        )
        return AgentState(
            goal=self.goal,
            plan_history=self.plan_history,
            step_executions=remaining + (execution,),
            memory_summary=self.memory_summary,
        )


def serialize_agent_state(state: AgentState) -> dict[str, Any]:
    """Convert validated domain state to the JSON-compatible graph state."""
    return {
        "goal": state.goal,
        "memory_summary": state.memory_summary,
        "plan_history": [
            {
                "revision": plan.revision,
                "goal": plan.goal,
                "steps": [
                    {
                        "id": step.id,
                        "intent": step.intent,
                        "completion_criterion": step.completion_criterion,
                    }
                    for step in plan.steps
                ],
            }
            for plan in state.plan_history
        ],
        "step_executions": [
            {
                "revision": execution.revision,
                "step_id": execution.step_id,
                "status": execution.status,
                "result": execution.result,
                "error": execution.error,
            }
            for execution in state.step_executions
        ],
    }


def deserialize_agent_state(payload: dict[str, Any]) -> AgentState:
    """Restore domain values and reapply all AgentState invariants."""
    plans = tuple(
        Plan(
            item["revision"],
            item["goal"],
            tuple(
                PlanStep(step["id"], step["intent"], step["completion_criterion"])
                for step in item["steps"]
            ),
        )
        for item in payload.get("plan_history", [])
    )
    executions = tuple(
        StepExecution(
            item["revision"],
            item["step_id"],
            item["status"],
            result=item.get("result"),
            error=item.get("error"),
        )
        for item in payload.get("step_executions", [])
    )
    return AgentState(
        payload["goal"],
        plan_history=plans,
        step_executions=executions,
        memory_summary=payload.get("memory_summary"),
    )
