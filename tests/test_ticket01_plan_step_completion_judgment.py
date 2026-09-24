import asyncio
from io import StringIO

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from agent.executor import Executor
from agent.trace import RunTrace
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
        self.tool_bindings = []

    def bind_tools(self, tools):
        self.tool_bindings.append(tuple(getattr(tool, "name", "") for tool in tools))
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


def _judge(completed: bool, evidence: str | None = None) -> AIMessage:
    if not completed:
        return AIMessage(content="BEGIN STEP_COMPLETION_PROGRESS\nALL_COMPLETED=false\nEND STEP_COMPLETION_PROGRESS")
    return AIMessage(content=(
        "BEGIN STEP_COMPLETION_PROGRESS\nALL_COMPLETED=true\n"
        "BEGIN COMPLETED_CRITERION\nNUMBER=1\n"
        f"EVIDENCE=\"{evidence}\"\nEND COMPLETED_CRITERION\n"
        "END STEP_COMPLETION_PROGRESS"
    ))


def test_executor_keeps_pending_after_judge_and_exposes_state_to_next_operational_request(monkeypatch):
    async def load_tools(_session):
        return [StructuredTool.from_function(_record, name="record")]

    monkeypatch.setattr("agent.executor.MultiServerMCPClient", _Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = _Model(
        AIMessage(content="", tool_calls=[{"name": "record", "args": {}, "id": "call-1"}]),
        _judge(False),
        AIMessage(content="Still checking", tool_calls=[{"name": "record", "args": {}, "id": "call-2"}]),
        _judge(True, "record returned artifact is available"),
        AIMessage(content="Published."),
    )

    async def run():
        async with Executor(model=model, max_rounds=3) as executor:
            return await executor.execute(_state(), 1, "publish")

    outcome = asyncio.run(run())

    assert outcome.execution.status == "completed"
    assert outcome.execution.result == "Published."
    assert outcome.execution.completion_evidence == "record returned artifact is available"
    assert "1. [pending] The artifact is available" in "\n".join(
        message.content for message in model.requests[2] if isinstance(message.content, str)
    )


def test_executor_uses_separate_tool_free_handoff_after_completed_judgment(monkeypatch):
    async def load_tools(_session):
        return [StructuredTool.from_function(_record, name="record")]

    monkeypatch.setattr("agent.executor.MultiServerMCPClient", _Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    output = StringIO()
    model = _Model(
        AIMessage(content="", tool_calls=[{"name": "record", "args": {}, "id": "call-1"}]),
        _judge(True, "record returned artifact is available"),
        AIMessage(content="Published."),
    )

    async def run():
        async with Executor(model=model, trace=RunTrace(output)) as executor:
            return await executor.execute(_state(), 1, "publish")

    outcome = asyncio.run(run())

    assert outcome.execution.status == "completed"
    assert outcome.execution.result == "Published."
    assert model.tool_bindings[-1] == ()
    assert "tool-free completion judge" in output.getvalue()
