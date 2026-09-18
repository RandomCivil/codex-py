import asyncio

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agent import RecoveryDecisionError
from agent.durable import DurableAgent
from memory.state import AgentState, ContextUpdate, ExecutionOutcome, Plan, PlanStep, StepContext, StepExecution, deserialize_agent_state


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

    async def execute(self, state, revision, step_id, step_context=None, *, recovery=False, attempt=None):
        self.calls += 1
        return ExecutionOutcome(
            StepExecution(revision, step_id, "completed", result="Inspected"),
            ContextUpdate(),
        )


class TwoStepPlanner:
    async def plan(self, state):
        return Plan(
            1,
            state.goal,
            (
                PlanStep("inspect", "Inspect", "Inspected"),
                PlanStep("change", "Change", "Changed"),
            ),
        )


class ContextRecordingExecutor(Executor):
    def __init__(self):
        super().__init__()
        self.contexts = []

    async def execute(self, state, revision, step_id, step_context=None, *, recovery=False, attempt=None):
        self.calls += 1
        self.contexts.append(step_context)
        if step_id == "inspect":
            return ExecutionOutcome(
                StepExecution(revision, step_id, "completed", result="Inspected"),
                ContextUpdate(
                    files_read=("README.md", "agent/durable.py"),
                    files_modified=("agent/durable.py",),
                    observations=("The durable graph owns the handoff.",),
                ),
            )
        return ExecutionOutcome(
            StepExecution(revision, step_id, "completed", result="Changed"),
            ContextUpdate(
                files_read=("agent/durable.py",),
                files_modified=("tests/test_durable.py",),
                observations=("The second step received prior context.",),
            ),
        )


def test_durable_agent_hands_successful_context_to_the_next_plan_step():
    executor = ContextRecordingExecutor()

    result = asyncio.run(
        DurableAgent(TwoStepPlanner(), executor, InMemorySaver()).run(
            "run-context", AgentState("Inspect repo")
        )
    )

    assert result["status"] == "completed"
    assert executor.contexts == [
        StepContext(),
        StepContext(
            files_read=("README.md", "agent/durable.py"),
            files_modified=("agent/durable.py",),
            observations=("revision 1, step inspect: The durable graph owns the handoff.",),
        ),
    ]


class ResumePlanner:
    def __init__(self):
        self.calls = 0

    async def plan(self, state):
        self.calls += 1
        revision = len(state.plan_history) + 1
        steps = (
            (PlanStep("inspect", "Inspect", "Inspected"), PlanStep("change", "Change", "Changed"))
            if revision == 1
            else (PlanStep("change", "Change", "Changed"),)
        )
        return Plan(
            revision,
            state.goal,
            steps,
        )


class FailOnSecondStepExecutor(Executor):
    def __init__(self):
        super().__init__()
        self.contexts = []

    async def execute(self, state, revision, step_id, step_context=None, *, recovery=False, attempt=None):
        self.calls += 1
        self.contexts.append(step_context)
        if self.calls == 1:
            return ExecutionOutcome(
                StepExecution(revision, step_id, "completed", result="Inspected"),
                ContextUpdate(
                    files_read=("README.md",),
                    files_modified=("agent/durable.py",),
                    observations=("The first step found the handoff point.",),
                ),
            )
        raise RuntimeError("process stopped")


class CompleteAfterResumeExecutor(Executor):
    def __init__(self):
        super().__init__()
        self.contexts = []

    async def execute(self, state, revision, step_id, step_context=None, *, recovery=False, attempt=None):
        self.calls += 1
        self.contexts.append(step_context)
        return ExecutionOutcome(
            StepExecution(revision, step_id, "completed", result="Changed"),
            ContextUpdate(),
        )


class FailureDoesNotAddContextExecutor(Executor):
    def __init__(self):
        super().__init__()
        self.contexts = []

    async def execute(self, state, revision, step_id, step_context=None, *, recovery=False, attempt=None):
        self.calls += 1
        self.contexts.append(step_context)
        if (revision, step_id) == (1, "inspect"):
            return ExecutionOutcome(
                StepExecution(revision, step_id, "completed", result="Inspected"),
                ContextUpdate(
                    files_read=("README.md",),
                    files_modified=("agent/durable.py",),
                    observations=("The first step found the handoff point.",),
                ),
            )
        if (revision, step_id) == (1, "change"):
            return ExecutionOutcome(
                StepExecution(revision, step_id, "failed", error="change failed")
            )
        return ExecutionOutcome(
            StepExecution(revision, step_id, "completed", result="Changed"),
            ContextUpdate(),
        )


def test_durable_agent_does_not_add_context_from_a_failed_execution():
    executor = FailureDoesNotAddContextExecutor()

    result = asyncio.run(
        DurableAgent(ResumePlanner(), executor, InMemorySaver()).run(
            "run-failed-context", AgentState("Inspect repo")
        )
    )

    prior_context = StepContext(
        files_read=("README.md",),
        files_modified=("agent/durable.py",),
        observations=("revision 1, step inspect: The first step found the handoff point.",),
    )
    assert result["status"] == "completed"
    assert executor.contexts == [StepContext(), prior_context, prior_context]


def test_durable_agent_preserves_only_successful_context_across_checkpoint_resume():
    saver = InMemorySaver()
    first_executor = FailOnSecondStepExecutor()
    with pytest.raises(RuntimeError, match="process stopped"):
        asyncio.run(
            DurableAgent(ResumePlanner(), first_executor, saver).run(
                "run-resume-context", AgentState("Inspect repo")
            )
        )

    resumed_executor = CompleteAfterResumeExecutor()
    resumed = asyncio.run(
        DurableAgent(ResumePlanner(), resumed_executor, saver).run(
            "run-resume-context", recovery="fail"
        )
    )

    assert resumed["status"] == "completed"
    assert resumed_executor.contexts == [
        StepContext(
            files_read=("README.md",),
            files_modified=("agent/durable.py",),
            observations=("revision 1, step inspect: The first step found the handoff point.",),
        )
    ]
    checkpoint = asyncio.run(
        DurableAgent(ResumePlanner(), Executor(), saver)._graph().aget_state(
            {"configurable": {"thread_id": "run-resume-context"}}
        )
    )
    assert checkpoint.values["files_read"] == ["README.md"]
    assert checkpoint.values["files_modified"] == ["agent/durable.py"]
    assert checkpoint.values["observations"] == [
        "revision 1, step inspect: The first step found the handoff point."
    ]


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
    async def execute(self, state, revision, step_id, step_context=None, *, recovery=False, attempt=None):
        self.calls += 1
        if self.calls == 1:
            return ExecutionOutcome(StepExecution(revision, step_id, "failed", error="inspection failed"))
        return ExecutionOutcome(
            StepExecution(revision, step_id, "completed", result="Inspected"),
            ContextUpdate(),
        )


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


def test_durable_agent_automatically_recovers_a_stale_running_step():
    saver = InMemorySaver()
    plan = Plan(1, "Inspect repo", (PlanStep("inspect", "Inspect", "Inspected"),))
    state = AgentState(
        "Inspect repo",
        plan_history=(plan,),
        step_executions=(StepExecution(1, "inspect", "running"),),
    )

    result = asyncio.run(DurableAgent(Planner(), Executor(), saver).run("run-3", state))

    assert result["status"] == "completed"


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
