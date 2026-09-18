"""Project-owned Agent-run metadata and exclusive execution leases."""

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping


LEASE_SECONDS = 60
RENEWAL_SECONDS = 15


class RunBusyError(RuntimeError):
    """Another owner currently holds the run lease."""


class ConfigurationMismatchError(RuntimeError):
    """The resume configuration differs from the run's saved configuration."""


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    thread_id: str
    status: str
    config_snapshot: dict[str, Any]
    config_fingerprint: str
    terminal_result: dict[str, Any] | None = None
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None


def configuration_snapshot(
    *,
    mcp: Mapping[str, Any],
    planner: Mapping[str, Any] | None = None,
    executor: Mapping[str, Any] | None = None,
    task_analyzer: Mapping[str, Any] | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Return effective non-secret component and execution configuration."""
    if planner is None:
        planner = {"base_url": base_url or "", "model_name": model or "", "response_format": "json_schema"}
    if executor is None:
        executor = planner
    if task_analyzer is None:
        task_analyzer = planner
    return {
        "planner": _provider_snapshot(planner),
        "executor": _provider_snapshot(executor),
        "task_analyzer": _provider_snapshot(task_analyzer),
        "mcp": _without_secrets(mcp),
    }


def _provider_snapshot(provider: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(provider, Mapping):
        provider = vars(provider)
    return {
        "base_url": provider["base_url"],
        "model_name": provider["model_name"],
        "response_format": provider["response_format"],
    }


def configuration_fingerprint(snapshot: Mapping[str, Any]) -> str:
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class InMemoryRunRegistry:
    """Small registry double for controlled Agent and CLI tests."""

    def __init__(self, *, server_now: Callable[[], datetime] | None = None) -> None:
        self.server_now = server_now or (lambda: datetime.now(timezone.utc))
        self._records: dict[str, RunRecord] = {}

    async def create(self, run_id: str, snapshot: Mapping[str, Any]) -> RunRecord:
        record = RunRecord(run_id, run_id, "running", dict(snapshot), configuration_fingerprint(snapshot))
        self._records[run_id] = record
        return record

    async def get(self, run_id: str) -> RunRecord:
        return self._records[run_id]

    async def ensure_compatible(self, run_id: str, snapshot: Mapping[str, Any]) -> RunRecord:
        record = await self.get(run_id)
        if record.config_fingerprint != configuration_fingerprint(snapshot):
            raise ConfigurationMismatchError("run configuration fingerprint does not match")
        return record

    async def acquire(self, run_id: str, owner: str) -> RunRecord:
        record = await self.get(run_id)
        now = self.server_now()
        if record.lease_owner and record.lease_owner != owner and record.lease_expires_at and record.lease_expires_at > now:
            raise RunBusyError("run busy")
        updated = RunRecord(**{**record.__dict__, "lease_owner": owner, "lease_expires_at": now + timedelta(seconds=LEASE_SECONDS)})
        self._records[run_id] = updated
        return updated

    async def renew(self, run_id: str, owner: str) -> RunRecord:
        record = await self.get(run_id)
        if record.lease_owner != owner or not record.lease_expires_at or record.lease_expires_at <= self.server_now():
            raise RunBusyError("run lease expired")
        updated = RunRecord(**{**record.__dict__, "lease_expires_at": self.server_now() + timedelta(seconds=LEASE_SECONDS)})
        self._records[run_id] = updated
        return updated

    async def release(self, run_id: str, owner: str) -> None:
        record = await self.get(run_id)
        if record.lease_owner == owner:
            self._records[run_id] = RunRecord(**{**record.__dict__, "lease_owner": None, "lease_expires_at": None})

    async def set_terminal(self, run_id: str, owner: str, result: Mapping[str, Any]) -> None:
        record = await self.get(run_id)
        if record.lease_owner != owner:
            raise RunBusyError("run lease expired")
        self._records[run_id] = RunRecord(**{**record.__dict__, "status": result["status"], "terminal_result": dict(result)})


class MySQLRunRegistry:
    """Run registry backed by the project-owned ``agent_runs`` table."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def create(self, run_id: str, snapshot: Mapping[str, Any]) -> RunRecord:
        await self._execute(
            "INSERT INTO agent_runs (run_id, thread_id, status, config_snapshot, config_fingerprint) VALUES (%s, %s, %s, %s, %s)",
            (run_id, run_id, "running", json.dumps(snapshot, sort_keys=True), configuration_fingerprint(snapshot)),
        )
        return await self.get(run_id)

    async def get(self, run_id: str) -> RunRecord:
        row = await self._fetchone(
            "SELECT run_id, thread_id, status, config_snapshot, config_fingerprint, terminal_result, lease_owner, lease_expires_at FROM agent_runs WHERE run_id=%s",
            (run_id,),
        )
        if row is None:
            raise KeyError(f"unknown Agent run: {run_id}")
        snapshot = row[3] if isinstance(row[3], dict) else json.loads(row[3])
        terminal = row[5] if isinstance(row[5], (dict, type(None))) else json.loads(row[5])
        return RunRecord(row[0], row[1], row[2], snapshot, row[4], terminal, row[6], _utc(row[7]))

    async def ensure_compatible(self, run_id: str, snapshot: Mapping[str, Any]) -> RunRecord:
        record = await self.get(run_id)
        if record.config_fingerprint != configuration_fingerprint(snapshot):
            raise ConfigurationMismatchError("run configuration fingerprint does not match")
        return record

    async def acquire(self, run_id: str, owner: str) -> RunRecord:
        changed = await self._execute(
            "UPDATE agent_runs SET lease_owner=%s, lease_expires_at=DATE_ADD(UTC_TIMESTAMP(6), INTERVAL 60 SECOND) WHERE run_id=%s AND (lease_owner IS NULL OR lease_owner=%s OR lease_expires_at <= UTC_TIMESTAMP(6))",
            (owner, run_id, owner),
        )
        if not changed:
            raise RunBusyError("run busy")
        return await self.get(run_id)

    async def renew(self, run_id: str, owner: str) -> RunRecord:
        changed = await self._execute(
            "UPDATE agent_runs SET lease_expires_at=DATE_ADD(UTC_TIMESTAMP(6), INTERVAL 60 SECOND) WHERE run_id=%s AND lease_owner=%s AND lease_expires_at > UTC_TIMESTAMP(6)",
            (run_id, owner),
        )
        if not changed:
            raise RunBusyError("run lease expired")
        return await self.get(run_id)

    async def release(self, run_id: str, owner: str) -> None:
        await self._execute("UPDATE agent_runs SET lease_owner=NULL, lease_expires_at=NULL WHERE run_id=%s AND lease_owner=%s", (run_id, owner))

    async def set_terminal(self, run_id: str, owner: str, result: Mapping[str, Any]) -> None:
        changed = await self._execute(
            "UPDATE agent_runs SET status=%s, terminal_result=%s WHERE run_id=%s AND lease_owner=%s",
            (result["status"], json.dumps(result, sort_keys=True), run_id, owner),
        )
        if not changed:
            raise RunBusyError("run lease expired")

    async def _fetchone(self, query: str, params: tuple[Any, ...]):
        async with self._pool.acquire() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(query, params)
                return await cursor.fetchone()

    async def _execute(self, query: str, params: tuple[Any, ...]) -> int:
        async with self._pool.acquire() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(query, params)
                changed = cursor.rowcount
            await connection.commit()
        return changed


# The public registry name describes the policy boundary; the implementation is MySQL-backed.
RunRegistry = MySQLRunRegistry


def _without_secrets(value: Any, key: str | None = None) -> Any:
    if key == "env" or (key and any(word in key.lower() for word in ("secret", "password", "token", "credential", "api_key"))):
        return None
    if isinstance(value, Mapping):
        return {str(item_key): cleaned for item_key, item in value.items() if (cleaned := _without_secrets(item, str(item_key))) is not None}
    if isinstance(value, (list, tuple)):
        return [_without_secrets(item) for item in value]
    return value


def _utc(value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value
