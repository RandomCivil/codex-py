import asyncio

from langgraph.checkpoint.memory import InMemorySaver

from agent.durable import DurableAgent
from agent.recovery import InMemoryStepRecoveryStore
from memory.state import AgentState, ContextUpdate, ExecutionOutcome, Plan, PlanStep, StepContext, StepExecution


def test_recovery_store_round_trips_immutable_attempt_history_and_trustworthy_context():
    store = InMemoryStepRecoveryStore()

    async def scenario():
        await store.record(
            "run-1",
            StepExecution(1, "inspect", "interrupted", error="owner stopped"),
            recovery=False,
            version=1,
        )
        await store.record(
            "run-1",
            StepExecution(1, "inspect", "completed", result="Inspected"),
            context_update=ContextUpdate(files_read=("README.md",), observations=("Found the entry point.",)),
            recovery=True,
            version=2,
        )

        snapshot = await store.restore("run-1", 2)
        assert [item.attempt for item in snapshot.attempts] == [1, 2]
        assert snapshot.latest_executions == (
            StepExecution(1, "inspect", "completed", result="Inspected"),
        )
        assert snapshot.context == StepContext(
            files_read=("README.md",), observations=("Found the entry point.",)
        )

    asyncio.run(scenario())


def test_recovery_store_rejects_unconfirmed_common_version_before_restore():
    store = InMemoryStepRecoveryStore()

    async def scenario():
        await store.record(
            "run-1",
            StepExecution(1, "inspect", "running"),
            version=3,
        )
        try:
            await store.restore("run-1", 2)
        except RuntimeError as error:
            assert "persistence version" in str(error)
        else:
            raise AssertionError("version mismatch must fail closed")

    asyncio.run(scenario())


def test_recovery_store_uses_common_version_not_step_id_to_order_history():
    store = InMemoryStepRecoveryStore()

    async def scenario():
        await store.record("run-1", StepExecution(1, "z-last", "completed", result="done"), context_update=ContextUpdate(), version=1)
        await store.record("run-1", StepExecution(1, "a-next", "running"), version=2)

        snapshot = await store.restore("run-1", 2)
        assert [item.version for item in snapshot.attempts] == [1, 2]
        assert snapshot.latest_executions[-1] == StepExecution(1, "a-next", "running")

    asyncio.run(scenario())


def test_durable_agent_records_running_and_completed_attempts_at_public_agent_boundary():
    class Planner:
        async def plan(self, state):
            return Plan(1, state.goal, (PlanStep("inspect", "Inspect", "Inspected"),))

    class Executor:
        calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def execute(self, state, revision, step_id, step_context=None, *, recovery=False, attempt=None):
            self.calls += 1
            return ExecutionOutcome(
                StepExecution(revision, step_id, "completed", result="Inspected"),
                ContextUpdate(files_read=("README.md",)),
            )

    async def scenario():
        store = InMemoryStepRecoveryStore()
        executor = Executor()
        result = await DurableAgent(
            Planner(), executor, InMemorySaver(), recovery_store=store
        ).run("run-2", AgentState("Inspect repo"))
        assert result["status"] == "completed"
        history = await store.history("run-2")
        assert [(item.attempt, item.execution.status) for item in history] == [
            (1, "running"),
            (1, "completed"),
        ]
        assert history[-1].context_update == ContextUpdate(files_read=("README.md",))

    asyncio.run(scenario())


def test_graceful_interrupt_immediately_records_the_interrupted_attempt():
    plan = Plan(1, "Inspect repo", (PlanStep("inspect", "Inspect", "Inspected"),))
    running = AgentState(
        "Inspect repo",
        plan_history=(plan,),
        step_executions=(StepExecution(1, "inspect", "running"),),
    )

    async def scenario():
        saver = InMemorySaver()
        store = InMemoryStepRecoveryStore()
        await store.record("run-3", running.step_executions[0], version=1)
        durable = DurableAgent(object(), object(), saver, recovery_store=store)
        await durable._graph().ainvoke(
            {"agent_state": running, "status": "interrupted", "persistence_version": 1},
            config={"configurable": {"thread_id": "run-3"}},
        )
        await durable.interrupt("run-3")
        return await store.history("run-3")

    history = asyncio.run(scenario())

    assert [(item.execution.status, item.version) for item in history] == [
        ("running", 1),
        ("interrupted", 2),
    ]
