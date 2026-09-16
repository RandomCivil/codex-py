import asyncio

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agent import RecoveryDecisionError
from agent.durable import DurableAgent
from memory.state import AgentState, Plan, PlanStep, StepExecution, deserialize_agent_state


class Planner:
    async def plan(self, state):
        return Plan(1, state.goal, (PlanStep("inspect", "Inspect", "Inspected"),))


class Executor:
    def __init__(self):
        self.calls = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def execute(self, state, revision, step_id):
        self.calls += 1
        return StepExecution(revision, step_id, "completed", result="Inspected")


class ReplanningPlanner:
    def __init__(self):
        self.calls = 0

    async def plan(self, state):
        self.calls += 1
        return Plan(
            self.calls,
            state.goal,
            (PlanStep("inspect", "Inspect", "Inspected"),),
        )


class FailOnceExecutor(Executor):
    async def execute(self, state, revision, step_id):
        self.calls += 1
        if self.calls == 1:
            return StepExecution(revision, step_id, "failed", error="inspection failed")
        return StepExecution(revision, step_id, "completed", result="Inspected")


def test_durable_agent_restores_a_terminal_run_without_executing_completed_work():
    saver = InMemorySaver()
    first_executor = Executor()
    first = asyncio.run(
        DurableAgent(Planner(), first_executor, saver).run("run-1", AgentState("Inspect repo"))
    )

    second_executor = Executor()
    resumed = asyncio.run(DurableAgent(Planner(), second_executor, saver).run("run-1"))

    assert first["status"] == "completed"
    assert resumed["status"] == "completed"
    assert deserialize_agent_state(resumed["state"]).step_executions[0].status == "completed"
    assert first_executor.calls == 1
    assert second_executor.calls == 0


def test_durable_agent_replans_after_failure_and_retains_immutable_history():
    saver = InMemorySaver()
    planner = ReplanningPlanner()
    executor = FailOnceExecutor()

    result = asyncio.run(
        DurableAgent(planner, executor, saver).run("run-2", AgentState("Inspect repo"))
    )

    state = deserialize_agent_state(result["state"])
    assert result["status"] == "completed"
    assert [plan.revision for plan in state.plan_history] == [1, 2]
    assert [(item.revision, item.status) for item in state.step_executions] == [(1, "failed"), (2, "completed")]


def test_durable_agent_requires_recovery_for_a_stale_running_step():
    saver = InMemorySaver()
    plan = Plan(1, "Inspect repo", (PlanStep("inspect", "Inspect", "Inspected"),))
    state = AgentState(
        "Inspect repo",
        plan_history=(plan,),
        step_executions=(StepExecution(1, "inspect", "running"),),
    )

    with pytest.raises(RecoveryDecisionError, match="requires --recovery"):
        asyncio.run(DurableAgent(Planner(), Executor(), saver).run("run-3", state))


def test_durable_abort_is_checkpointed_as_a_terminal_result():
    saver = InMemorySaver()
    plan = Plan(1, "Inspect repo", (PlanStep("inspect", "Inspect", "Inspected"),))
    state = AgentState(
        "Inspect repo",
        plan_history=(plan,),
        step_executions=(StepExecution(1, "inspect", "running"),),
    )

    aborted = asyncio.run(DurableAgent(Planner(), Executor(), saver).run("run-4", state, recovery="abort"))
    resumed_executor = Executor()
    resumed = asyncio.run(DurableAgent(Planner(), resumed_executor, saver).run("run-4"))

    assert aborted["status"] == resumed["status"] == "blocked"
    assert deserialize_agent_state(resumed["state"]).step_executions[0].status == "interrupted"
    assert resumed_executor.calls == 0
