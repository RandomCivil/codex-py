import asyncio
import json

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from agent.executor import Executor
from memory.state import AgentState, Plan, PlanStep


class _Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class _Client:
    def __init__(self, _config):
        pass

    def session(self, _name):
        return _Session()


class _Model:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages):
        self.requests.append(list(messages))
        return next(self.responses)


def _state():
    plan = Plan(1, "Publish", (PlanStep("publish", "Publish", "The artifact is available"),))
    return AgentState("Publish", plan_history=(plan,))


def _record() -> str:
    """Create the artifact."""
    return "artifact is available"


def _judge(completed: bool = True) -> AIMessage:
    if not completed:
        return AIMessage(content="BEGIN STEP_COMPLETION_PROGRESS\nALL_COMPLETED=false\nEND STEP_COMPLETION_PROGRESS")
    return AIMessage(content=(
        "BEGIN STEP_COMPLETION_PROGRESS\nALL_COMPLETED=true\n"
        "BEGIN COMPLETED_CRITERION\nNUMBER=1\n"
        'EVIDENCE="artifact is available"\nEND COMPLETED_CRITERION\n'
        "END STEP_COMPLETION_PROGRESS"
    ))


def test_mixed_tool_batch_judges_only_success_and_returns_failure_for_correction(monkeypatch):
    async def load_tools(_session):
        def fail():
            """Fail the requested operation."""
            raise RuntimeError("permission denied")

        return [
            StructuredTool.from_function(_record, name="record"),
            StructuredTool.from_function(fail, name="fail"),
        ]

    monkeypatch.setattr("agent.executor.MultiServerMCPClient", _Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = _Model(
        AIMessage(content="", tool_calls=[
            {"name": "record", "args": {}, "id": "success"},
            {"name": "fail", "args": {}, "id": "failed"},
        ]),
        _judge(False),
        AIMessage(content="Still checking."),
        _judge(False),
    )

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(_state(), 1, "publish")

    outcome = asyncio.run(run())

    assert outcome.execution.status == "failed", outcome.execution.error
    assert model.requests, outcome.execution.error
    judge_request = model.requests[1]
    judge_text = "\n".join(getattr(message, "content", "") for message in judge_request)
    assert "artifact is available" in judge_text
    judge_evidence = json.loads(judge_request[-1].content)
    assert "permission denied" not in json.dumps(judge_evidence["successful_tool_results"])
    assert "permission denied" in judge_evidence["failed_tool_results"][0]["error"]
    correction_request = next(
        request for request in model.requests
        if any(getattr(message, "status", None) == "error" for message in request)
    )
    correction_text = "\n".join(getattr(message, "content", "") for message in correction_request)
    assert "permission denied" in correction_text


def test_all_failed_tool_batch_does_not_invoke_completion_judge(monkeypatch):
    def fail():
        """Fail the requested operation."""
        raise RuntimeError("unavailable")

    async def load_tools(_session):
        return [StructuredTool.from_function(fail, name="fail")]

    monkeypatch.setattr("agent.executor.MultiServerMCPClient", _Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = _Model(
        AIMessage(content="", tool_calls=[{"name": "fail", "args": {}, "id": "failed"}]),
        AIMessage(content="No progress."),
        _judge(False),
    )

    async def run():
        async with Executor(model=model, max_rounds=1) as executor:
            return await executor.execute(_state(), 1, "publish")

    outcome = asyncio.run(run())

    assert outcome.execution.status == "failed"
    assert len(model.requests) == 1


class _BudgetFailurePolicy:
    class _Context:
        def as_messages(self, **_kwargs):
            return []

    def __init__(self):
        self.maintains = 0

    def fresh_for_invocation(self):
        return self

    def record_model_use(self):
        pass

    async def maintain(self, _durable_state):
        self.maintains += 1
        if self.maintains == 2:
            raise RuntimeError("required Runtime context exceeds its budget")
        return self._Context()

    async def record_tool_round(self, *_args, **_kwargs):
        pass


def test_settled_evidence_that_cannot_fit_runtime_context_fails_the_step(monkeypatch):
    async def load_tools(_session):
        return [StructuredTool.from_function(_record, name="record")]

    monkeypatch.setattr("agent.executor.MultiServerMCPClient", _Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = _Model(
        AIMessage(content="", tool_calls=[{"name": "record", "args": {}, "id": "call-1"}]),
        _judge(True),
    )
    policy = _BudgetFailurePolicy()

    async def run():
        async with Executor(model=model, context_policy=policy) as executor:
            return await executor.execute(_state(), 1, "publish")

    outcome = asyncio.run(run())

    assert outcome.execution.status == "failed"
    assert "Runtime context exceeds its budget" in outcome.execution.error


def test_invalid_judgment_repair_is_attempted_once_and_failed_repair_keeps_progress_pending(monkeypatch):
    async def load_tools(_session):
        return [StructuredTool.from_function(_record, name="record")]

    monkeypatch.setattr("agent.executor.MultiServerMCPClient", _Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    malformed = AIMessage(content="not a STEP_COMPLETION_PROGRESS block")
    model = _Model(
        AIMessage(content="", tool_calls=[{"name": "record", "args": {}, "id": "call-1"}]),
        malformed,
        malformed,
        AIMessage(content="Try again."),
        malformed,
        malformed,
    )

    async def run():
        async with Executor(model=model, max_rounds=2) as executor:
            return await executor.execute(_state(), 1, "publish")

    outcome = asyncio.run(run())

    assert outcome.execution.status == "failed"
    assert outcome.execution.completion_evidence is None
    assert len(model.requests) == 6
    repair_text = "\n".join(getattr(message, "content", "") for message in model.requests[2])
    assert "Validation error:" in repair_text
    assert "not a STEP_COMPLETION_PROGRESS block" in repair_text
    assert "[pending] The artifact is available" in model.requests[3][-1].content


def test_completion_judge_provider_failure_fails_the_step(monkeypatch):
    async def load_tools(_session):
        return [StructuredTool.from_function(_record, name="record")]

    monkeypatch.setattr("agent.executor.MultiServerMCPClient", _Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)

    class ProviderFailureModel(_Model):
        async def ainvoke(self, messages):
            if len(self.requests) == 1:
                raise RuntimeError("judge provider unavailable")
            return await super().ainvoke(messages)

    model = ProviderFailureModel(
        AIMessage(content="", tool_calls=[{"name": "record", "args": {}, "id": "call-1"}]),
    )

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(_state(), 1, "publish")

    outcome = asyncio.run(run())

    assert outcome.execution.status == "failed"
    assert "judge provider unavailable" in outcome.execution.error
