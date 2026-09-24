"""Project-owned, immutable Step recovery records.

The LangGraph checkpointer remains the scheduling authority.  This module is
the small recovery-facing projection used to rebuild the latest domain state
and trustworthy context without reading saver-owned tables.
"""

from dataclasses import dataclass
import json
from typing import Any

from memory.state import ContextUpdate, StepContext, StepExecution


class PersistenceVersionMismatchError(RuntimeError):
    """The checkpoint and recovery projection cannot be reconciled safely."""


@dataclass(frozen=True, slots=True)
class RecoveryAttempt:
    run_id: str
    revision: int
    step_id: str
    attempt: int
    execution: StepExecution
    context_update: ContextUpdate | None
    recovery: bool
    version: int


@dataclass(frozen=True, slots=True)
class RecoverySnapshot:
    attempts: tuple[RecoveryAttempt, ...]
    latest_executions: tuple[StepExecution, ...]
    context: StepContext
    version: int


class InMemoryStepRecoveryStore:
    """Controlled store for Agent tests; it follows the production contract."""

    def __init__(self) -> None:
        self._attempts: list[RecoveryAttempt] = []

    async def record(
        self,
        run_id: str,
        execution: StepExecution,
        *,
        context_update: ContextUpdate | None = None,
        recovery: bool = False,
        version: int,
    ) -> RecoveryAttempt:
        prior = [
            item
            for item in self._attempts
            if item.run_id == run_id
            and (item.revision, item.step_id) == (execution.revision, execution.step_id)
        ]
        latest = prior[-1] if prior else None
        attempt = (
            latest.attempt
            if latest is not None and latest.execution.status in {"pending", "running"}
            else len({item.attempt for item in prior}) + 1
        )
        durable_execution = _durable_execution(execution)
        record = RecoveryAttempt(
            run_id,
            durable_execution.revision,
            durable_execution.step_id,
            attempt,
            durable_execution,
            context_update,
            recovery,
            version,
        )
        self._attempts.append(record)
        return record

    async def restore(self, run_id: str, checkpoint_version: int) -> RecoverySnapshot:
        attempts = tuple(
            sorted(
                (item for item in self._attempts if item.run_id == run_id),
                key=lambda item: item.version,
            )
        )
        if not attempts:
            return RecoverySnapshot((), (), StepContext(), checkpoint_version)
        if attempts[-1].version != checkpoint_version:
            raise PersistenceVersionMismatchError(
                "checkpoint and recovery records have no common persistence version"
            )
        return _snapshot(attempts, checkpoint_version)

    async def history(self, run_id: str) -> tuple[RecoveryAttempt, ...]:
        return tuple(item for item in self._attempts if item.run_id == run_id)


class MySQLStepRecoveryStore:
    """MySQL adapter for the project-owned recovery-record table."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def record(
        self,
        run_id: str,
        execution: StepExecution,
        *,
        context_update: ContextUpdate | None = None,
        recovery: bool = False,
        version: int,
    ) -> RecoveryAttempt:
        async with self._pool.acquire() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT attempt, status FROM step_recovery_attempts
                    WHERE run_id=%s AND revision=%s AND step_id=%s
                    ORDER BY record_id DESC LIMIT 1
                    """,
                    (run_id, execution.revision, execution.step_id),
                )
                latest = await cursor.fetchone()
                attempt = (
                    latest[0]
                    if latest and latest[1] in {"pending", "running"}
                    else (latest[0] + 1 if latest else 1)
                )
                await cursor.execute(
                    """
                    INSERT INTO step_recovery_attempts
                    (run_id, revision, step_id, attempt, status, result, error,
                     completion_evidence, context_update, recovery, common_version)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    (
                        run_id,
                        execution.revision,
                        execution.step_id,
                        attempt,
                        execution.status,
                        execution.result,
                        execution.error,
                        _durable_execution(execution).completion_evidence,
                        _context_json(context_update),
                        recovery,
                        version,
                    ),
                )
            await connection.commit()
        return RecoveryAttempt(
            run_id,
            execution.revision,
            execution.step_id,
            attempt,
            _durable_execution(execution),
            context_update,
            recovery,
            version,
        )

    async def restore(self, run_id: str, checkpoint_version: int) -> RecoverySnapshot:
        async with self._pool.acquire() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    """
                    SELECT revision, step_id, attempt, status, result, error,
                           completion_evidence, context_update, recovery, common_version
                    FROM step_recovery_attempts WHERE run_id=%s
                    ORDER BY common_version, record_id
                    """,
                    (run_id,),
                )
                rows = await cursor.fetchall()
        attempts = tuple(_row_to_attempt(run_id, row) for row in rows)
        if attempts and attempts[-1].version != checkpoint_version:
            raise PersistenceVersionMismatchError(
                "checkpoint and recovery records have no common persistence version"
            )
        return _snapshot(attempts, checkpoint_version)

    async def history(self, run_id: str) -> tuple[RecoveryAttempt, ...]:
        snapshot = await self.restore(run_id, await self.latest_version(run_id))
        return snapshot.attempts

    async def latest_version(self, run_id: str) -> int:
        async with self._pool.acquire() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "SELECT COALESCE(MAX(common_version), 0) FROM step_recovery_attempts WHERE run_id=%s",
                    (run_id,),
                )
                return (await cursor.fetchone())[0]


def _snapshot(attempts: tuple[RecoveryAttempt, ...], version: int) -> RecoverySnapshot:
    latest: dict[tuple[int, str], StepExecution] = {}
    files_read: list[str] = []
    files_modified: list[str] = []
    observations: list[str] = []
    for item in attempts:
        latest[(item.revision, item.step_id)] = item.execution
        if item.execution.status != "completed" or item.context_update is None:
            continue
        update = item.context_update
        _append_unique(files_read, update.files_read)
        _append_unique(files_modified, update.files_modified)
        _append_unique(observations, update.observations)
    return RecoverySnapshot(
        attempts,
        tuple(latest.values()),
        StepContext(tuple(files_read), tuple(files_modified), tuple(observations)),
        version,
    )


def _append_unique(target: list[str], values: tuple[str, ...]) -> None:
    for value in values:
        if value not in target:
            target.append(value)


def _context_json(update: ContextUpdate | None) -> str | None:
    if update is None:
        return None
    return json.dumps({
        "files_read": list(update.files_read),
        "files_modified": list(update.files_modified),
        "observations": list(update.observations),
    })


def _row_to_attempt(run_id: str, row: tuple[Any, ...]) -> RecoveryAttempt:
    if len(row) == 9:
        revision, step_id, attempt, status, result, error, raw_context, recovery, version = row
        completion_evidence = None
    else:
        revision, step_id, attempt, status, result, error, completion_evidence, raw_context, recovery, version = row
    context = json.loads(raw_context) if raw_context else None
    update = ContextUpdate(**context) if context else None
    return RecoveryAttempt(
        run_id,
        revision,
        step_id,
        attempt,
        _durable_execution(
            StepExecution(
                revision,
                step_id,
                status,
                result=result,
                error=error,
                completion_evidence=completion_evidence,
            )
        ),
        update,
        bool(recovery),
        version,
    )


def _durable_execution(execution: StepExecution) -> StepExecution:
    """Drop attempt-local judge evidence unless the Step is finally complete."""
    if execution.status == "completed":
        return execution
    if execution.completion_evidence is None:
        return execution
    return StepExecution(
        execution.revision,
        execution.step_id,
        execution.status,
        result=execution.result,
        error=execution.error,
    )
