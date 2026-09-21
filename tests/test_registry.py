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


def test_effective_component_snapshot_excludes_keys_and_allows_key_rotation():
    first = configuration_snapshot(
        planner={"base_url": "https://planner.test", "model_name": "planner", "api_key": "one"},
        executor={"base_url": "https://executor.test", "model_name": "executor", "api_key": "two"},
        mcp={"command": "atom"},
    )
    rotated = configuration_snapshot(
        planner={"base_url": "https://planner.test", "model_name": "planner", "api_key": "rotated"},
        executor={"base_url": "https://executor.test", "model_name": "executor", "api_key": "rotated"},
        mcp={"command": "atom"},
    )

    assert first["planner"] == {"base_url": "https://planner.test", "model_name": "planner"}
    assert first["executor"] == {"base_url": "https://executor.test", "model_name": "executor"}
    assert first["task_analyzer"] == first["planner"]
    assert "api_key" not in str(first)
    assert configuration_fingerprint(first) == configuration_fingerprint(rotated)


def test_runtime_context_snapshot_excludes_credentials_and_tracks_effective_configuration():
    original = configuration_snapshot(
        planner={"base_url": "https://planner.test", "model_name": "planner"},
        runtime_context={
            "base_url": "https://context.test",
            "model_name": "context-model",
            "api_key": "secret",
        },
        mcp={},
    )
    changed = configuration_snapshot(
        planner={"base_url": "https://planner.test", "model_name": "planner"},
        runtime_context={
            "base_url": "https://context.test",
            "model_name": "changed-context-model",
            "api_key": "rotated-secret",
        },
        mcp={},
    )

    assert original["runtime_context"] == {
        "base_url": "https://context.test",
        "model_name": "context-model",
    }
    assert "api_key" not in str(original)
    assert configuration_fingerprint(original) != configuration_fingerprint(changed)


@pytest.mark.parametrize(
    "change",
    [
        {"base_url": "https://changed.test"},
        {"model_name": "changed-model"},
    ],
)
def test_effective_component_drift_changes_resume_fingerprint(change):
    original = configuration_snapshot(
        planner={"base_url": "https://planner.test", "model_name": "planner"},
        executor={"base_url": "https://executor.test", "model_name": "executor"},
        mcp={},
    )
    changed_executor = {"base_url": "https://executor.test", "model_name": "executor", **change}
    changed = configuration_snapshot(
        planner={"base_url": "https://planner.test", "model_name": "planner"},
        executor=changed_executor,
        mcp={},
    )

    assert configuration_fingerprint(original) != configuration_fingerprint(changed)
