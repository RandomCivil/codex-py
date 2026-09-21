import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

from agent.conversation import (
    ConversationInput,
    ConversationService,
    ConversationTurn,
    InMemoryConversationStore,
    _MySQLDurableAgent,
)
from agent.execution import ExecutionAnswer, create_execution_mode
from agent.durable import DurableConversationRunner


class DurableRun:
    def __init__(self, status):
        self.status = status
        self.calls = []

    async def run(self, run_id, state, **kwargs):
        self.calls.append((run_id, state, kwargs))
        return {"run_id": run_id, "status": self.status, "state": None}


def test_mysql_durable_adapter_delegates_plan_execute_run(monkeypatch):
    calls = []

    async def run(*args, **kwargs):
        calls.append((args, kwargs))
        return {"run_id": args[1], "status": "blocked", "state": None}

    monkeypatch.setattr("agent.durable._run", run)
    adapter = _MySQLDurableAgent(
        "mysql://database",
        configuration=object(),
        cwd="../project",
        log_level="error",
    )
    conversation_input = ConversationInput(
        "550e8400-e29b-41d4-a716-446655440000",
        (),
        "Ship it",
    )

    result = asyncio.run(
        adapter.run(
            "550e8400-e29b-41d4-a716-446655440001",
            state=type("State", (), {"goal": "Ship it"})(),
            conversation_input=conversation_input,
        )
    )

    assert result["status"] == "blocked"
    assert calls[0][0] == (
        "mysql://database",
        "550e8400-e29b-41d4-a716-446655440001",
        "Ship it",
    )
    assert calls[0][1]["cwd"] == "../project"
    assert calls[0][1]["execution_mode"] == "plan_execute"
    assert calls[0][1]["conversation_input"] is conversation_input


def test_mysql_durable_adapter_restores_tool_cwd_for_recovery(monkeypatch):
    calls = []

    async def run(*args, **kwargs):
        calls.append((args, kwargs))
        return {"run_id": args[1], "status": "blocked", "state": None}

    @asynccontextmanager
    async def connection_pool(_configuration):
        yield object()

    class Registry:
        def __init__(self, _pool):
            pass

        async def get(self, _run_id):
            return SimpleNamespace(config_snapshot={"mcp": {"tool_cwd": "/saved/project"}})

    monkeypatch.setattr("agent.durable._run", run)
    monkeypatch.setattr("agent.migration._connection_pool", connection_pool)
    monkeypatch.setattr("agent.migration._parse_url", lambda url: url)
    monkeypatch.setattr("agent.conversation.MySQLRunRegistry", Registry)
    adapter = _MySQLDurableAgent(
        "mysql://database",
        configuration=object(),
        cwd=None,
        restore_cwd=True,
    )

    asyncio.run(adapter.run("550e8400-e29b-41d4-a716-446655440001"))

    assert calls[0][1]["cwd"] == "/saved/project"


def test_plan_execute_preserves_blocked_result_inside_an_independent_conversation_run():
    durable = DurableRun("blocked")
    run_ids = []

    def runner_factory(mode, run_id):
        run_ids.append((mode, run_id))
        return create_execution_mode("plan_execute", durable_agent=durable, run_id=run_id)

    service = ConversationService(InMemoryConversationStore(), runner_factory)

    result = asyncio.run(service.run(user_input="Ship it", execution_mode="plan_execute"))

    assert result.status == "blocked"
    assert result.run_id == run_ids[0][1]
    assert durable.calls[0][0] == result.run_id
    assert durable.calls[0][2]["conversation_input"].current_input == "Ship it"


class RecoverableRunner:
    def __init__(self, answer):
        self.answer = answer
        self.recovery = []

    async def resume(self, recovery=None):
        self.recovery.append(recovery)
        return self.answer


def test_conversation_resume_reconciles_the_active_durable_turn():
    store = InMemoryConversationStore()
    conv_id = "550e8400-e29b-41d4-a716-446655440000"
    run_id = "550e8400-e29b-41d4-a716-446655440001"
    store.append(ConversationTurn(conv_id, 1, run_id, "plan_execute", "Ship it", "running"))
    runner = RecoverableRunner(ExecutionAnswer("Published", "completed"))
    service = ConversationService(
        store,
        lambda mode, selected_run_id: runner,
        recovery_factory=lambda selected_run_id: runner,
    )

    result = asyncio.run(service.resume(conv_id=conv_id, recovery="fail"))

    assert result.status == "completed"
    assert runner.recovery == ["fail"]
    assert store.get(conv_id).turns[0].status == "completed"


def test_durable_conversation_runner_uses_the_turn_run_for_run_and_recovery():
    durable = DurableRun("blocked")
    runner = DurableConversationRunner(durable, "550e8400-e29b-41d4-a716-446655440002")

    result = asyncio.run(runner.run(type("Input", (), {"history": (), "current_input": "Ship it"})()))

    assert result == ExecutionAnswer(None, "blocked", "plan-execute run blocked")
    assert durable.calls[0][0] == "550e8400-e29b-41d4-a716-446655440002"
    assert durable.calls[0][1].goal == "Ship it"
    assert durable.calls[0][2]["conversation_input"].current_input == "Ship it"


def test_durable_conversation_runner_reports_a_safe_actionable_provider_status():
    class ProviderFailure(RuntimeError):
        status_code = 402

    class FailingDurable:
        async def run(self, *_args, **_kwargs):
            raise ProviderFailure("provider payload that must not reach the caller")

    answer = asyncio.run(
        DurableConversationRunner(FailingDurable(), "550e8400-e29b-41d4-a716-446655440002").run(
            type("Input", (), {"history": (), "current_input": "Ship it"})()
        )
    )

    assert answer == ExecutionAnswer(
        None,
        "failed",
        "plan-execute execution failed: provider returned HTTP 402 (Insufficient Balance)",
    )


def test_next_conversation_turn_reconciles_a_standalone_completed_agent_run():
    store = InMemoryConversationStore()
    conv_id = "550e8400-e29b-41d4-a716-446655440003"
    run_id = "550e8400-e29b-41d4-a716-446655440004"
    store.append(ConversationTurn(conv_id, 1, run_id, "plan_execute", "First", "running"))
    runner = RecoverableRunner(ExecutionAnswer("Second", "completed"))
    service = ConversationService(
        store,
        lambda mode, selected_run_id: runner,
        run_result_factory=lambda selected_run_id: {
            "status": "completed",
            "answer": "First done",
            "error": None,
        },
    )

    result = asyncio.run(service.run(conv_id=conv_id, user_input="Second", execution_mode="direct"))

    assert result.sequence == 2
    assert store.get(conv_id).turns[0].status == "completed"
    assert runner.recovery == []
