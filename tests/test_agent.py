import asyncio

import pytest

from agent import Agent, AgentResult, PlanningValidationError, RecoveryDecisionError
from memory.state import AgentState, ContextUpdate, ExecutionOutcome, Plan, PlanStep, StepExecution


class ControlledPlanner:
    def __init__(self, plan):
        self.plan_to_return = plan
        self.states = []

    async def plan(self, state):
        self.states.append(state)
        return self.plan_to_return


class ControlledExecutor:
    def __init__(self, executions=None):
        self.entered = 0
        self.exited = 0
        self.calls = []
        self.executions = executions or {}

    async def __aenter__(self):
        self.entered += 1
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.exited += 1

    async def execute(self, state, revision, step_id):
        self.calls.append((state, revision, step_id))
        execution = self.executions.get(
            (revision, step_id),
            StepExecution(revision, step_id, "completed", result=f"{step_id} done"),
        )
        return ExecutionOutcome(
            execution,
            ContextUpdate() if execution.status == "completed" else None,
        )


def test_agent_derives_and_completes_a_plan_through_one_run():
    plan = Plan(1, "Prepare release", (PlanStep("inspect", "Inspect", "Risks listed"),))
    planner = ControlledPlanner(plan)
    executor = ControlledExecutor()

    result = asyncio.run(Agent(planner, executor).run(AgentState("Prepare release")))

    assert result.status == "completed"
    assert result.state.plan_history == (plan,)
    assert result.state.step_executions == (
        StepExecution(1, "inspect", "completed", result="inspect done"),
    )
    assert planner.states == [AgentState("Prepare release")]
    assert executor.calls[0][1:] == (1, "inspect")
    assert (executor.entered, executor.exited) == (1, 1)


def test_agent_executes_plan_steps_in_declared_order_and_records_each_result():
    plan = Plan(
        1,
        "Prepare release",
        (
            PlanStep("inspect", "Inspect", "Risks listed"),
            PlanStep("publish", "Publish", "Release available"),
        ),
    )
    planner = ControlledPlanner(plan)
    executor = ControlledExecutor()

    result = asyncio.run(Agent(planner, executor).run(AgentState("Prepare release")))

    assert [call[2] for call in executor.calls] == ["inspect", "publish"]
    assert executor.calls[1][0].step_executions == (
        StepExecution(1, "inspect", "completed", result="inspect done"),
    )
    assert [execution.step_id for execution in result.state.step_executions] == [
        "inspect",
        "publish",
    ]


def test_agent_returns_an_already_completed_plan_without_new_work():
    plan = Plan(1, "Prepare release", (PlanStep("inspect", "Inspect", "Risks listed"),))
    state = AgentState(
        "Prepare release",
        plan_history=(plan,),
        step_executions=(StepExecution(1, "inspect", "completed", result="already done"),),
    )

    class PlannerMustNotRun:
        async def plan(self, state):
            raise AssertionError("completed state should not be planned again")

    class ExecutorMustNotRun(ControlledExecutor):
        async def execute(self, state, revision, step_id):
            raise AssertionError("completed step should not execute again")

    executor = ExecutorMustNotRun()
    result = asyncio.run(Agent(PlannerMustNotRun(), executor).run(state))

    assert result.status == "completed"
    assert result.state == state
    assert (executor.entered, executor.exited) == (1, 1)


def test_agent_records_failure_stops_revision_and_replans_from_failed_state():
    first_plan = Plan(
        1,
        "Prepare release",
        (
            PlanStep("inspect", "Inspect", "Risks listed"),
            PlanStep("publish", "Publish", "Release available"),
        ),
    )
    second_plan = Plan(
        2,
        "Prepare release",
        (PlanStep("recover", "Recover", "Release available"),),
    )

    class SequentialPlanner:
        def __init__(self):
            self.states = []

        async def plan(self, state):
            self.states.append(state)
            return (first_plan, second_plan)[len(self.states) - 1]

    planner = SequentialPlanner()
    executor = ControlledExecutor(
        {(1, "inspect"): StepExecution(1, "inspect", "failed", error="risk scan failed")}
    )

    result = asyncio.run(Agent(planner, executor).run(AgentState("Prepare release")))

    assert [call[1:] for call in executor.calls] == [(1, "inspect"), (2, "recover")]
    assert planner.states[1].step_executions == (
        StepExecution(1, "inspect", "failed", error="risk scan failed"),
    )
    assert result.state.plan_history == (first_plan, second_plan)
    assert result.state.step_executions == (
        StepExecution(1, "inspect", "failed", error="risk scan failed"),
        StepExecution(2, "recover", "completed", result="recover done"),
    )


def test_agent_blocks_after_third_failed_revision_without_requesting_a_fourth_plan():
    plans = tuple(
        Plan(revision, "Prepare release", (PlanStep("inspect", "Inspect", "Risks listed"),))
        for revision in (1, 2, 3)
    )

    class SequentialPlanner:
        def __init__(self):
            self.states = []

        async def plan(self, state):
            self.states.append(state)
            return plans[len(self.states) - 1]

    class AlwaysFailingExecutor(ControlledExecutor):
        async def execute(self, state, revision, step_id):
            self.calls.append((state, revision, step_id))
            return ExecutionOutcome(StepExecution(revision, step_id, "failed", error=f"revision {revision} failed"))

    planner = SequentialPlanner()
    executor = AlwaysFailingExecutor()

    result = asyncio.run(Agent(planner, executor).run(AgentState("Prepare release")))

    assert result.status == "blocked"
    assert result.state.plan_history == plans
    assert result.state.step_executions == tuple(
        StepExecution(revision, "inspect", "failed", error=f"revision {revision} failed")
        for revision in (1, 2, 3)
    )
    assert len(planner.states) == 3
    assert [state.plan_history for state in planner.states] == [
        (),
        (plans[0],),
        (plans[0], plans[1]),
    ]


def test_agent_replans_before_executing_when_current_revision_has_a_recorded_failure():
    first_plan = Plan(
        1,
        "Prepare release",
        (
            PlanStep("inspect", "Inspect", "Risks listed"),
            PlanStep("publish", "Publish", "Release available"),
        ),
    )
    second_plan = Plan(
        2,
        "Prepare release",
        (PlanStep("recover", "Recover", "Release available"),),
    )
    state = AgentState(
        "Prepare release",
        plan_history=(first_plan,),
        step_executions=(StepExecution(1, "publish", "failed", error="publish failed"),),
    )

    class SequentialPlanner:
        def __init__(self):
            self.states = []

        async def plan(self, state):
            self.states.append(state)
            return second_plan

    planner = SequentialPlanner()
    executor = ControlledExecutor()

    result = asyncio.run(Agent(planner, executor).run(state))

    assert [call[1:] for call in executor.calls] == [(2, "recover")]
    assert planner.states == [state]
    assert result.status == "completed"


def test_agent_blocks_on_a_recorded_failure_in_the_third_revision():
    plans = tuple(
        Plan(revision, "Prepare release", (PlanStep("inspect", "Inspect", "Risks listed"),))
        for revision in (1, 2, 3)
    )
    state = AgentState(
        "Prepare release",
        plan_history=plans,
        step_executions=(
            StepExecution(3, "inspect", "failed", error="third revision failed"),
        ),
    )

    class PlannerMustNotRun:
        async def plan(self, state):
            raise AssertionError("a fourth plan must not be requested")

    executor = ControlledExecutor()
    result = asyncio.run(Agent(PlannerMustNotRun(), executor).run(state))

    assert result == AgentResult("blocked", state)
    assert executor.calls == []
    assert (executor.entered, executor.exited) == (1, 1)


def test_agent_fails_an_interrupted_step_into_immutable_replanning():
    first_plan = Plan(1, "Prepare release", (PlanStep("publish", "Publish", "Released"),))
    second_plan = Plan(2, "Prepare release", (PlanStep("verify", "Verify", "Verified"),))
    state = AgentState(
        "Prepare release",
        plan_history=(first_plan,),
        step_executions=(StepExecution(1, "publish", "interrupted", error="owner stopped"),),
    )

    class Planner:
        async def plan(self, current):
            assert current.step_executions == (
                StepExecution(1, "publish", "failed", error="owner stopped"),
            )
            return second_plan

    result = asyncio.run(Agent(Planner(), ControlledExecutor()).run(state, recovery="fail"))

    assert result.status == "completed"
    assert result.state.plan_history == (first_plan, second_plan)
    assert result.state.step_executions[0].status == "failed"


def test_agent_aborts_an_interrupted_step_without_executing_more_work():
    plan = Plan(1, "Prepare release", (PlanStep("publish", "Publish", "Released"),))
    state = AgentState(
        "Prepare release",
        plan_history=(plan,),
        step_executions=(StepExecution(1, "publish", "interrupted", error="uncertain side effect"),),
    )
    executor = ControlledExecutor()

    result = asyncio.run(Agent(object(), executor).run(state, recovery="abort"))

    assert result == AgentResult("blocked", state)
    assert executor.calls == []


def test_agent_rejects_retry_for_an_interrupted_non_idempotent_step():
    plan = Plan(1, "Prepare release", (PlanStep("publish", "Publish", "Released"),))
    state = AgentState(
        "Prepare release",
        plan_history=(plan,),
        step_executions=(StepExecution(1, "publish", "interrupted"),),
    )

    with pytest.raises(RecoveryDecisionError, match="retry"):
        asyncio.run(Agent(object(), ControlledExecutor()).run(state, recovery="retry"))


def test_agent_requires_recovery_for_a_running_step_instead_of_reexecuting_it():
    plan = Plan(
        1,
        "Prepare release",
        tuple(
            PlanStep(step_id, step_id.title(), f"{step_id} done")
            for step_id in ("completed", "pending", "running", "skipped")
        ),
    )
    state = AgentState(
        "Prepare release",
        plan_history=(plan,),
        step_executions=(
            StepExecution(1, "completed", "completed", result="already done"),
            StepExecution(1, "pending", "pending"),
            StepExecution(1, "running", "running"),
            StepExecution(1, "skipped", "skipped"),
        ),
    )
    executor = ControlledExecutor()

    with pytest.raises(RecoveryDecisionError, match="requires"):
        asyncio.run(Agent(object(), executor).run(state))
    assert executor.calls == []


def test_agent_propagates_planner_validation_errors_without_recording_execution():
    error = PlanningValidationError("invalid plan contract")

    class FailingPlanner:
        async def plan(self, state):
            raise error

    executor = ControlledExecutor()

    with pytest.raises(PlanningValidationError) as raised:
        asyncio.run(Agent(FailingPlanner(), executor).run(AgentState("Prepare release")))

    assert raised.value is error
    assert (executor.entered, executor.exited) == (1, 1)
    assert executor.calls == []
