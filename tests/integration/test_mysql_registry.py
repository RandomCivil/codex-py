import asyncio
import os
import uuid

import pytest

from agent.migration import _connection_pool, _parse_url, migrate_database
from agent.registry import ConfigurationMismatchError, MySQLRunRegistry, RunBusyError
from agent.durable import DurableAgent, _mysql_saver
from agent import RecoveryDecisionError
from memory.state import AgentState, Plan, PlanStep, StepExecution, deserialize_agent_state, serialize_agent_state


MYSQL_URL = os.environ.get("CODEX_TEST_MYSQL_URL")


@pytest.mark.integration
@pytest.mark.skipif(
    MYSQL_URL is None,
    reason="CODEX_TEST_MYSQL_URL is not configured; skipping MySQL integration test",
)
def test_mysql_registry_enforces_lease_and_configuration_compatibility():
    run_id = str(uuid.uuid4())

    async def scenario():
        async with _connection_pool(_parse_url(MYSQL_URL)) as pool:
            registry = MySQLRunRegistry(pool)
            await registry.create(run_id, {"model": "model-a"})
            await registry.acquire(run_id, "owner-a")
            with pytest.raises(RunBusyError):
                await registry.acquire(run_id, "owner-b")
            with pytest.raises(ConfigurationMismatchError):
                await registry.ensure_compatible(run_id, {"model": "model-b"})
            async with pool.acquire() as connection:
                async with connection.cursor() as cursor:
                    await cursor.execute(
                        "UPDATE agent_runs SET lease_expires_at=DATE_SUB(UTC_TIMESTAMP(6), INTERVAL 1 SECOND) WHERE run_id=%s",
                        (run_id,),
                    )
                await connection.commit()
            record = await registry.acquire(run_id, "owner-b")
            assert record.lease_owner == "owner-b"

    migrate_database(MYSQL_URL)
    asyncio.run(scenario())


@pytest.mark.integration
@pytest.mark.skipif(
    MYSQL_URL is None,
    reason="CODEX_TEST_MYSQL_URL is not configured; skipping MySQL integration test",
)
def test_mysql_checkpoint_resumes_terminal_work_in_a_fresh_application_instance():
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

    run_id = str(uuid.uuid4())

    async def scenario():
        async with _mysql_saver(MYSQL_URL) as saver:
            first_executor = Executor()
            first = await DurableAgent(Planner(), first_executor, saver).run(run_id, AgentState("Inspect repo"))
        async with _mysql_saver(MYSQL_URL) as saver:
            resumed_executor = Executor()
            resumed = await DurableAgent(Planner(), resumed_executor, saver).run(run_id)
        assert first["status"] == resumed["status"] == "completed"
        assert deserialize_agent_state(resumed["state"]).step_executions[0].status == "completed"
        assert first_executor.calls == 1
        assert resumed_executor.calls == 0

    migrate_database(MYSQL_URL)
    asyncio.run(scenario())


@pytest.mark.integration
@pytest.mark.skipif(
    MYSQL_URL is None,
    reason="CODEX_TEST_MYSQL_URL is not configured; skipping MySQL integration test",
)
def test_mysql_checkpoint_refuses_stale_running_work_without_recovery():
    class Planner:
        async def plan(self, state):
            raise AssertionError("stale work must require a recovery decision")

    class Executor:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def execute(self, state, revision, step_id):
            raise AssertionError("stale work must not execute")

    run_id = str(uuid.uuid4())
    plan = Plan(1, "Inspect repo", (PlanStep("inspect", "Inspect", "Inspected"),))
    state = AgentState("Inspect repo", plan_history=(plan,), step_executions=(StepExecution(1, "inspect", "running"),))

    async def scenario():
        async with _mysql_saver(MYSQL_URL) as saver:
            durable = DurableAgent(Planner(), Executor(), saver)
            await durable._graph().ainvoke(
                {"agent_state": serialize_agent_state(state), "status": "interrupted"},
                config={"configurable": {"thread_id": run_id}},
            )
        async with _mysql_saver(MYSQL_URL) as saver:
            with pytest.raises(RecoveryDecisionError, match="requires --recovery"):
                await DurableAgent(Planner(), Executor(), saver).run(run_id)

    migrate_database(MYSQL_URL)
    asyncio.run(scenario())
