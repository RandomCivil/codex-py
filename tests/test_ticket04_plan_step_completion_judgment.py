import asyncio

from langgraph.checkpoint.memory import InMemorySaver

from agent.durable import DurableAgent
from agent.recovery import InMemoryStepRecoveryStore, MySQLStepRecoveryStore
from memory.state import (
    AgentState,
    ContextUpdate,
    ExecutionOutcome,
    Plan,
    PlanStep,
    StepExecution,
    deserialize_agent_state,
    serialize_agent_state,
)


def _state(status="pending", *, evidence=None):
    plan = Plan(1, "Publish", (PlanStep("publish", "Publish", "Artifact is available"),))
    return AgentState(
        "Publish",
        plan_history=(plan,),
        step_executions=(
            StepExecution(1, "publish", status, result="handoff", completion_evidence=evidence),
        ),
    )


def test_only_completed_step_serialization_retains_completion_evidence():
    pending = serialize_agent_state(_state(evidence="not yet confirmed"))
    failed = serialize_agent_state(_state("failed", evidence="provisional proof"))
    completed = serialize_agent_state(_state("completed", evidence="artifact URL responds"))

    assert "completion_evidence" not in pending["step_executions"][0]
    assert "completion_evidence" not in failed["step_executions"][0]
    assert completed["step_executions"][0]["completion_evidence"] == "artifact URL responds"


def test_recovery_store_round_trips_final_evidence_and_reads_legacy_rows():
    store = InMemoryStepRecoveryStore()

    async def scenario():
        await store.record(
            "run-1",
            StepExecution(1, "publish", "completed", result="Published", completion_evidence="URL responds"),
            context_update=ContextUpdate(observations=("published",)),
            version=1,
        )
        await store.record(
            "run-1",
            StepExecution(1, "other", "failed", error="stopped", completion_evidence="unconfirmed"),
            version=2,
        )

        snapshot = await store.restore("run-1", 2)

        class Cursor:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return None

            async def execute(self, query, params):
                return None

            async def fetchall(self):
                # A pre-migration row has no completion_evidence column.
                return [(1, "publish", 1, "completed", "Published", None, None, False, 1)]

        class Connection:
            def cursor(self):
                return Cursor()

        class Acquire:
            async def __aenter__(self):
                return Connection()

            async def __aexit__(self, *args):
                return None

        class Pool:
            def acquire(self):
                return Acquire()

        legacy = await MySQLStepRecoveryStore(Pool()).restore("run-legacy", 1)
        return snapshot, legacy

    snapshot, legacy = asyncio.run(scenario())
    assert snapshot.latest_executions[0].completion_evidence == "URL responds"
    assert snapshot.latest_executions[1].completion_evidence is None
    assert legacy.latest_executions[0].completion_evidence is None


def test_completed_step_is_restored_with_evidence_and_not_executed_again():
    class Planner:
        async def plan(self, state):
            return Plan(1, state.goal, (PlanStep("publish", "Publish", "Artifact is available"),))

    class Executor:
        def __init__(self):
            self.calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def execute(self, *args, **kwargs):
            self.calls += 1
            return ExecutionOutcome(
                StepExecution(1, "publish", "completed", result="Published", completion_evidence="URL responds"),
                ContextUpdate(),
            )

    async def scenario():
        saver = InMemorySaver()
        store = InMemoryStepRecoveryStore()
        first_executor = Executor()
        first = await DurableAgent(Planner(), first_executor, saver, recovery_store=store).run(
            "run-completed", AgentState("Publish")
        )
        second_executor = Executor()
        second = await DurableAgent(Planner(), second_executor, saver, recovery_store=store).run(
            "run-completed"
        )
        return first, second, first_executor, second_executor

    first, second, first_executor, second_executor = asyncio.run(scenario())
    assert first["status"] == second["status"] == "completed"
    restored = deserialize_agent_state(second["state"])
    assert restored.step_executions[0].completion_evidence == "URL responds"
    assert first_executor.calls == 1
    assert second_executor.calls == 0


def test_interrupted_step_recovery_starts_without_prior_completion_evidence():
    class Planner:
        async def plan(self, state):
            return Plan(1, state.goal, (PlanStep("publish", "Publish", "Artifact is available"),))

    class Executor:
        def __init__(self, fail=False):
            self.fail = fail
            self.calls = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def execute(self, state, revision, step_id, *args, **kwargs):
            self.calls.append((state.step_executions[-1], kwargs.get("recovery"), kwargs.get("attempt")))
            if self.fail:
                raise RuntimeError("process stopped")
            return ExecutionOutcome(
                StepExecution(revision, step_id, "completed", result="Published", completion_evidence="fresh URL responds"),
                ContextUpdate(),
            )

    async def scenario():
        saver = InMemorySaver()
        store = InMemoryStepRecoveryStore()
        first_executor = Executor(fail=True)
        try:
            await DurableAgent(Planner(), first_executor, saver, recovery_store=store).run(
                "run-interrupted", AgentState("Publish")
            )
        except RuntimeError:
            pass
        resumed_executor = Executor()
        result = await DurableAgent(Planner(), resumed_executor, saver, recovery_store=store).run(
            "run-interrupted"
        )
        return result, resumed_executor, await store.history("run-interrupted")

    result, resumed_executor, history = asyncio.run(scenario())
    assert result["status"] == "completed"
    assert resumed_executor.calls[0][0].status == "running"
    assert resumed_executor.calls[0][0].completion_evidence is None
    assert resumed_executor.calls[0][1:] == (True, 2)
    assert deserialize_agent_state(result["state"]).step_executions[0].completion_evidence == "fresh URL responds"
    assert [(item.attempt, item.execution.status) for item in history] == [
        (1, "running"),
        (1, "interrupted"),
        (2, "running"),
        (2, "completed"),
    ]
