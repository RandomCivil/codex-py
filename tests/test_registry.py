import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from agent.registry import (
    ConfigurationMismatchError,
    InMemoryRunRegistry,
    RunBusyError,
    configuration_fingerprint,
    configuration_snapshot,
)


def test_configuration_fingerprint_is_stable_and_excludes_credentials():
    first = configuration_snapshot(
        base_url="https://llm.example/v1",
        model="model-a",
        mcp={"command": "atom", "args": ["--stdio"], "api_key": "secret"},
    )
    second = configuration_snapshot(
        base_url="https://llm.example/v1",
        model="model-a",
        mcp={"api_key": "other-secret", "args": ["--stdio"], "command": "atom"},
    )

    assert first == second
    assert "api_key" not in first["mcp"]
    assert configuration_fingerprint(first) == configuration_fingerprint(second)


def test_registry_allows_takeover_only_after_server_time_lease_expiry():
    now = datetime(2026, 9, 16, 12, tzinfo=timezone.utc)
    registry = InMemoryRunRegistry(server_now=lambda: now)

    async def scenario():
        await registry.create("run-1", {"model": "model-a"})
        await registry.acquire("run-1", "owner-a")
        with pytest.raises(RunBusyError):
            await registry.acquire("run-1", "owner-b")

        registry.server_now = lambda: now + timedelta(seconds=61)
        record = await registry.acquire("run-1", "owner-b")
        return record.lease_owner

    assert asyncio.run(scenario()) == "owner-b"


def test_registry_renewal_extends_the_lease_from_server_time():
    now = datetime(2026, 9, 16, 12, tzinfo=timezone.utc)
    registry = InMemoryRunRegistry(server_now=lambda: now)

    async def scenario():
        await registry.create("run-1", {"model": "model-a"})
        await registry.acquire("run-1", "owner-a")
        registry.server_now = lambda: now + timedelta(seconds=15)
        record = await registry.renew("run-1", "owner-a")
        return record.lease_expires_at

    assert asyncio.run(scenario()) == now + timedelta(seconds=75)


def test_resume_rejects_configuration_drift_before_runner_is_called():
    registry = InMemoryRunRegistry()
    calls = []

    async def scenario():
        await registry.create("run-1", {"model": "model-a"})
        try:
            await registry.ensure_compatible("run-1", {"model": "model-b"})
        except ConfigurationMismatchError:
            return
        calls.append("runner")

    asyncio.run(scenario())
    assert calls == []
