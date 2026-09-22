import asyncio
import json
import uuid
from types import SimpleNamespace

import pytest

from agent.conversation import (
    ConversationBusyError,
    ConversationNotFoundError,
    ConversationService,
    ConversationTurn,
    InMemoryConversationStore,
    ConversationInput,
    _select_conversation_mode,
)
from agent.execution import ExecutionAnswer
from agent.execution import DirectMode
from agent.planner import PlanningValidationError
from agent.registry import ConfigurationMismatchError, RunBusyError


def _task_analysis_protocol(values):
    return "\n".join([
        "BEGIN TASK_ANALYSIS",
        *[f"{key.upper()}={json.dumps(value)}" for key, value in values.items()],
        "END TASK_ANALYSIS",
    ])


class ControlledRunner:
    def __init__(self, answer=ExecutionAnswer("hello", "completed")):
        self.answer = answer
        self.inputs = []
        self.store = None

    async def run(self, conversation_input):
        self.inputs.append(conversation_input)
        assert self.store.get(conversation_input.conv_id).turns[0].status == "running"
        return self.answer


def test_first_direct_turn_creates_conversation_before_running_the_mode():
    store = InMemoryConversationStore()
    runner = ControlledRunner()
    runner.store = store
    service = ConversationService(store, lambda mode, run_id: runner)

    result = asyncio.run(service.run(user_input="Hello", execution_mode="direct"))

    assert uuid.UUID(result.conv_id).version == 4
    assert uuid.UUID(result.run_id).version == 4
    assert result.sequence == 1
    assert result.execution_mode == "direct"
    assert result.status == "completed"
    assert result.answer == "hello"
    assert runner.inputs[0].history == ()
    assert runner.inputs[0].current_input == "Hello"
    assert store.get(result.conv_id).turns[0].user_input == "Hello"


def test_task_analyzer_selects_the_persisted_mode_before_running_the_turn():
    store = InMemoryConversationStore()
    runner = ControlledRunner()
    runner.store = store
    selected = []

    async def select_mode(conversation_input, run_id):
        selected.append((conversation_input, run_id))
        return "react"

    result = asyncio.run(
        ConversationService(
            store,
            lambda mode, run_id: runner,
            mode_selector=select_mode,
        ).run(user_input="Investigate this")
    )

    assert result.execution_mode == "react"
    assert store.get(result.conv_id).turns[0].execution_mode == "react"
    assert selected[0][0].current_input == "Investigate this"


def test_conversation_task_analyzer_is_traced_before_routing(monkeypatch, capsys):
    created = []
    analysis = {
        "task_type": "retrieval",
        "goal_clarity": 1.0,
        "needs_tools": True,
        "expected_steps": 3,
        "expected_horizon": "medium",
        "reasoning_summary": "inspect project files",
    }

    class Model:
        def __init__(self, *args, **kwargs):
            self.on_request = kwargs["on_request"]
            self.on_event = kwargs["on_event"]
            created.append(self)

        async def stream_text(self, goal, **kwargs):
            self.on_request({"model": "model", "input": goal})
            self.on_event(SimpleNamespace(type="response.created"))
            document = _task_analysis_protocol(analysis)
            self.on_event(SimpleNamespace(type="response.output_text.delta", delta=document))
            self.on_event(
                SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(
                        usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
                    ),
                )
            )
            yield document

        async def close(self):
            pass

    monkeypatch.setattr("llm.LLM", Model)
    configuration = SimpleNamespace(
        task_analyzer=SimpleNamespace(
            base_url="https://provider.test/v1",
            api_key="secret",
            model_name="model",
        )
    )

    mode = asyncio.run(
        _select_conversation_mode(
            ConversationInput("conv", (), "项目有哪些flutter组件"),
            "run-123",
            configuration=configuration,
            log_level="error",
        )
    )

    lines = capsys.readouterr().err.splitlines()
    assert mode == "react"
    assert len(created) == 1
    assert any("component=task_analyzer" in line for line in lines)
    assert lines[-1] == "[run-123] [task route] mode=react"


def test_conversation_task_analyzer_failure_traces_plan_execute_fallback(monkeypatch, capsys):
    class FailingModel:
        def __init__(self, *args, **kwargs):
            pass

        async def stream_text(self, goal, **kwargs):
            raise RuntimeError("provider failure")
            yield

        async def close(self):
            pass

    monkeypatch.setattr("llm.LLM", FailingModel)
    configuration = SimpleNamespace(
        task_analyzer=SimpleNamespace(
            base_url="https://provider.test/v1",
            api_key="secret",
            model_name="model",
        )
    )

    mode = asyncio.run(
        _select_conversation_mode(
            ConversationInput("conv", (), "Inspect project"),
            "run-123",
            configuration=configuration,
            log_level="error",
        )
    )

    assert mode == "plan_execute"
    assert capsys.readouterr().err.splitlines()[-1] == (
        "[run-123] [task route] mode=plan_execute reason=task_analysis_failed"
    )


def test_continuation_receives_public_history_and_current_input_separately():
    store = InMemoryConversationStore()
    runner = ControlledRunner()
    runner.store = store
    service = ConversationService(store, lambda mode, run_id: runner)

    first = asyncio.run(service.run(user_input="First", execution_mode="direct"))
    second = asyncio.run(
        service.run(conv_id=first.conv_id, user_input="Second", execution_mode="direct")
    )

    assert second.sequence == 2
    assert runner.inputs[-1].current_input == "Second"
    assert runner.inputs[-1].history == (
        {
            "sequence": 1,
            "user_input": "First",
            "execution_mode": "direct",
            "run_id": first.run_id,
            "status": "completed",
            "answer": "hello",
            "error": None,
        },
    )


def test_direct_mode_renders_conversation_input_as_structured_model_input():
    class Model:
        async def stream_text(self, value, **kwargs):
            self.value = value
            yield 'BEGIN ANSWER\nTEXT="answer"\nEND ANSWER'

    model = Model()
    conversation_input = type(
        "Input",
        (),
        {
            "history": ({"sequence": 1, "status": "completed", "answer": "old"},),
            "current_input": "new",
        },
    )()

    answer = asyncio.run(DirectMode(model).run(conversation_input))

    assert answer == ExecutionAnswer("answer", "completed")
    assert model.value == '{"history":[{"sequence":1,"status":"completed","answer":"old"}],"current_input":"new"}'


def test_conversation_exposes_a_safe_planning_validation_failure():
    class PlannerFailureRunner:
        async def run(self, _conversation_input):
            raise PlanningValidationError("planning output is invalid")

    result = asyncio.run(
        ConversationService(
            InMemoryConversationStore(), lambda _mode, _run_id: PlannerFailureRunner()
        ).run(user_input="Plan this", execution_mode="plan_execute")
    )

    assert result.status == "failed"
    assert result.error == "planning failed: planning output is invalid"


def test_append_to_unknown_conversation_is_rejected_without_creating_one():
    service = ConversationService(InMemoryConversationStore(), lambda mode, run_id: ControlledRunner())

    with pytest.raises(ConversationNotFoundError):
        asyncio.run(service.run(conv_id=str(uuid.uuid4()), user_input="Hello"))


def test_create_allows_a_chat_generated_conversation_id():
    store = InMemoryConversationStore()
    runner = ControlledRunner()
    runner.store = store
    service = ConversationService(store, lambda mode, run_id: runner)
    conv_id = "550e8400-e29b-41d4-a716-446655440000"

    result = asyncio.run(
        service.run(
            conv_id=conv_id,
            user_input="Hello",
            execution_mode="direct",
            create=True,
        )
    )

    assert result.conv_id == conv_id
    assert result.sequence == 1
    assert store.get(conv_id).turns[0].user_input == "Hello"


def test_second_append_reports_the_active_turn_as_busy():
    started = asyncio.Event()
    release = asyncio.Event()

    class BlockingRunner(ControlledRunner):
        async def run(self, conversation_input):
            started.set()
            await release.wait()
            return self.answer

    store = InMemoryConversationStore()
    service = ConversationService(store, lambda mode, run_id: BlockingRunner())

    async def exercise():
        first = asyncio.create_task(service.run(user_input="First"))
        await started.wait()
        active = store.get(next(iter(store.conversations))).turns[0]
        with pytest.raises(ConversationBusyError) as error:
            await service.run(conv_id=active.conv_id, user_input="Second")
        release.set()
        await first
        return error.value

    error = asyncio.run(exercise())
    assert error.sequence == 1
    assert error.run_id == next(iter(store.conversations.values())).turns[0].run_id


@pytest.mark.parametrize("error", [RunBusyError("run busy"), ConfigurationMismatchError("configuration changed")])
def test_conversation_resume_keeps_active_turn_when_recovery_cannot_start(error):
    store = InMemoryConversationStore()
    conv_id = "550e8400-e29b-41d4-a716-446655440000"
    store.append(ConversationTurn(conv_id, 1, "550e8400-e29b-41d4-a716-446655440001", "plan_execute", "Ship it", "running"))

    class Runner:
        async def resume(self, recovery):
            raise error

    service = ConversationService(store, lambda *args: Runner(), recovery_factory=lambda *args: Runner())

    with pytest.raises(type(error)):
        asyncio.run(service.resume(conv_id=conv_id))

    assert store.get(conv_id).turns[0].status == "running"
