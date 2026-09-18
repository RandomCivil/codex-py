import asyncio
import os
import uuid

import pytest

from agent.migration import migrate_database
from agent.recovery import MySQLStepRecoveryStore, PersistenceVersionMismatchError
from agent.migration import _connection_pool, _parse_url
from memory.state import ContextUpdate, StepExecution


MYSQL_URL = os.environ.get("CODEX_TEST_MYSQL_URL")


@pytest.mark.integration
@pytest.mark.skipif(
    MYSQL_URL is None,
    reason="CODEX_TEST_MYSQL_URL is not configured; skipping MySQL integration test",
)
def test_mysql_recovery_attempts_round_trip_and_fail_closed_on_version_mismatch():
    run_id = str(uuid.uuid4())

    async def scenario():
        async with _connection_pool(_parse_url(MYSQL_URL)) as pool:
            store = MySQLStepRecoveryStore(pool)
            await store.record(
                run_id,
                StepExecution(1, "inspect", "running"),
                version=1,
            )
            await store.record(
                run_id,
                StepExecution(1, "inspect", "completed", result="Inspected"),
                context_update=ContextUpdate(files_read=("README.md",)),
                version=2,
            )
            restored = await store.restore(run_id, 2)
            assert [item.attempt for item in restored.attempts] == [1, 1]
            assert restored.latest_executions == (
                StepExecution(1, "inspect", "completed", result="Inspected"),
            )
            assert restored.context.files_read == ("README.md",)
            with pytest.raises(PersistenceVersionMismatchError):
                await store.restore(run_id, 1)

    migrate_database(MYSQL_URL)
    asyncio.run(scenario())
