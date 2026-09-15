from dataclasses import FrozenInstanceError

import pytest

from memory.state import AgentState, Plan, PlanStep, StepExecution


def test_plan_is_an_immutable_ordered_whole_goal_contract():
    first = PlanStep("inspect", "Inspect the repository", "Relevant files are identified")
    second = PlanStep("change", "Apply the change", "The requested behavior is implemented")

    plan = Plan(1, "Implement the feature", (first, second))

    assert plan.revision == 1
    assert plan.goal == "Implement the feature"
    assert plan.steps == (first, second)
    with pytest.raises(FrozenInstanceError):
        plan.steps = ()


def test_plan_rejects_empty_or_duplicate_step_ids():
    with pytest.raises(ValueError, match="at least one"):
        Plan(1, "A goal", ())

    step = PlanStep("same", "one", "done")
    with pytest.raises(ValueError, match="unique"):
        Plan(1, "A goal", (step, PlanStep("same", "two", "done")))


def test_agent_state_keeps_plan_history_and_step_execution_separate():
    plan = Plan(1, "A goal", (PlanStep("one", "Do one", "One is done"),))
    execution = StepExecution(1, "one", "completed", result="finished")
    state = AgentState("A goal", plan_history=(plan,), step_executions=(execution,))

    assert state.plan_history == (plan,)
    assert state.step_executions == (execution,)
    assert state.step_executions[0].status == "completed"
    with pytest.raises(FrozenInstanceError):
        state.memory_summary = "changed"


def test_agent_state_accepts_a_new_sequential_revision_without_mutating_history():
    first = Plan(1, "A goal", (PlanStep("one", "Do one", "One is done"),))
    second = Plan(2, "A goal", (PlanStep("two", "Do two", "Two is done"),))
    state = AgentState("A goal", plan_history=(first,))

    updated = state.with_plan(second)

    assert state.plan_history == (first,)
    assert updated.plan_history == (first, second)


def test_agent_state_records_execution_without_changing_the_plan():
    plan = Plan(1, "A goal", (PlanStep("one", "Do one", "One is done"),))
    state = AgentState("A goal", plan_history=(plan,))

    updated = state.with_step_execution(StepExecution(1, "one", "running"))

    assert state.step_executions == ()
    assert updated.plan_history == (plan,)
    assert updated.step_executions == (StepExecution(1, "one", "running"),)

    completed = updated.with_step_execution(StepExecution(1, "one", "completed", result="finished"))
    assert completed.step_executions == (StepExecution(1, "one", "completed", result="finished"),)


@pytest.mark.parametrize("revision", [0, 4])
def test_plan_rejects_revision_outside_the_three_revision_budget(revision):
    with pytest.raises(ValueError, match="revision"):
        Plan(revision, "A goal", (PlanStep("one", "Do one", "One is done"),))


def test_agent_state_rejects_invalid_history_and_execution_associations():
    first = Plan(1, "A goal", (PlanStep("one", "Do one", "One is done"),))
    third = Plan(3, "A goal", (PlanStep("three", "Do three", "Three is done"),))
    with pytest.raises(ValueError, match="sequential"):
        AgentState("A goal", plan_history=(first, third))

    with pytest.raises(ValueError, match="associated"):
        AgentState("A goal", plan_history=(first,), step_executions=(StepExecution(1, "missing", "pending"),))


def test_step_execution_exposes_only_supported_statuses():
    with pytest.raises(ValueError, match="status"):
        StepExecution(1, "one", "done")
