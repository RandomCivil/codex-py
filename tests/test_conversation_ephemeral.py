import asyncio

import pytest

from agent.conversation import (
    ConversationService,
    ConversationTurn,
    InMemoryConversationStore,
    _make_ephemeral_runner,
)
from agent.execution import ExecutionAnswer


def test_ephemeral_runner_forwards_selected_working_directory_to_tool_runtime(monkeypatch, tmp_path):
    captured = []

    def make_mode(mode, **kwargs):
        captured.append((mode, kwargs))
        return object()

    monkeypatch.setattr("agent.conversation.create_execution_mode", make_mode)

    _make_ephemeral_runner(
        "react",
        "run-1",
        configuration=object(),
        cwd=str(tmp_path / "project"),
        log_level="error",
    )

    assert captured[0][1]["tool_runtime"]._tool_cwd == str(tmp_path / "project")


def test_interrupted_ephemeral_turn_is_failed_and_can_be_followed_by_a_new_turn():
    store = InMemoryConversationStore()
    started = asyncio.Event()
    release = asyncio.Event()
    inputs = []

    class BlockingRunner:
        async def run(self, conversation_input):
            inputs.append(conversation_input)
            started.set()
            await release.wait()
            return ExecutionAnswer("unexpected", "completed")

    class CompletedRunner:
        async def run(self, conversation_input):
            inputs.append(conversation_input)
            return ExecutionAnswer("continued", "completed")

    runners = iter((BlockingRunner(), CompletedRunner()))
    service = ConversationService(store, lambda mode, run_id: next(runners))

    async def exercise():
        first_task = asyncio.create_task(
            service.run(user_input="First", execution_mode="tool_agent")
        )
        await started.wait()
        first = store.get(next(iter(store.conversations))).turns[0]

        first_task.cancel()
        try:
            await first_task
        except asyncio.CancelledError:
            pass

        assert store.get(first.conv_id).turns[0].status == "failed"

        second = await service.run(
            conv_id=first.conv_id,
            user_input="Second",
            execution_mode="react",
        )
        return first, second

    first, second = asyncio.run(exercise())

    assert second.sequence == 2
    assert second.status == "completed"
    assert inputs[-1].history[0]["status"] == "failed"
    assert inputs[-1].current_input == "Second"


@pytest.mark.parametrize("execution_mode", ["tool_agent", "react"])
def test_ephemeral_mode_selection_receives_complete_public_history_and_new_input(execution_mode):
    store = InMemoryConversationStore()
    inputs = []

    class Runner:
        def __init__(self, answer):
            self.answer = answer

        async def run(self, conversation_input):
            inputs.append(conversation_input)
            return self.answer

    runners = iter(
        (
            Runner(ExecutionAnswer("first", "completed")),
            Runner(ExecutionAnswer(None, "failed", error="tool failed")),
            Runner(ExecutionAnswer("third", "completed")),
        )
    )
    selected_modes = []

    def make_runner(mode, run_id):
        selected_modes.append(mode)
        return next(runners)

    service = ConversationService(store, make_runner)

    async def exercise():
        first = await service.run(user_input="First", execution_mode=execution_mode)
        second = await service.run(
            conv_id=first.conv_id,
            user_input="Second",
            execution_mode=execution_mode,
        )
        third = await service.run(
            conv_id=first.conv_id,
            user_input="Third",
            execution_mode=execution_mode,
        )
        return first, second, third

    first, second, third = asyncio.run(exercise())

    assert selected_modes == [execution_mode] * 3
    assert second.status == "failed"
    assert third.status == "completed"
    assert inputs[1].history == (store.get(first.conv_id).turns[0].public(),)
    assert inputs[1].current_input == "Second"
    assert inputs[2].history == (
        store.get(first.conv_id).turns[0].public(),
        store.get(first.conv_id).turns[1].public(),
    )
    assert inputs[2].current_input == "Third"


def test_unrecoverable_active_ephemeral_turn_is_failed_before_next_append():
    store = InMemoryConversationStore()
    conv_id = "11111111-1111-4111-8111-111111111111"
    run_id = "22222222-2222-4222-8222-222222222222"
    store.append(
        ConversationTurn(
            conv_id,
            1,
            run_id,
            "tool_agent",
            "stale input",
            "running",
        )
    )

    class Runner:
        async def run(self, conversation_input):
            return ExecutionAnswer("continued", "completed")

    service = ConversationService(store, lambda mode, run_id: Runner())

    result = asyncio.run(
        service.run(conv_id=conv_id, user_input="Continue", execution_mode="react")
    )

    assert result.sequence == 2
    assert store.get(conv_id).turns[0].status == "failed"
    assert store.get(conv_id).turns[0].error == "ephemeral conversation turn could not be recovered"


def test_restarting_with_run_reconciliation_does_not_query_registry_for_ephemeral_turn():
    store = InMemoryConversationStore()
    conv_id = "11111111-1111-4111-8111-111111111111"
    store.append(
        ConversationTurn(
            conv_id,
            1,
            "22222222-2222-4222-8222-222222222222",
            "react",
            "stale input",
            "running",
        )
    )

    async def missing_agent_run(run_id):
        raise AssertionError("ephemeral turns must not query the Agent-run registry")

    class Runner:
        async def run(self, conversation_input):
            return ExecutionAnswer("continued", "completed")

    service = ConversationService(
        store,
        lambda mode, run_id: Runner(),
        run_result_factory=missing_agent_run,
    )

    result = asyncio.run(
        service.run(conv_id=conv_id, user_input="Continue", execution_mode="direct")
    )

    assert result.sequence == 2
    assert store.get(conv_id).turns[0].status == "failed"
    assert store.get(conv_id).turns[0].error == "ephemeral conversation turn could not be recovered"
