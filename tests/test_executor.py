import asyncio

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from agent.executor import Executor
from memory.state import AgentState, Plan, PlanStep


class Session:
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass


class Client:
    def __init__(self, config): self.config = config
    def session(self, name): return Session()


class Model:
    def __init__(self, *responses): self.responses = iter(responses); self.bound_tools = None; self.requests = []
    def bind_tools(self, tools): self.bound_tools = tools; return self
    async def ainvoke(self, messages):
        self.requests.append(messages)
        return next(self.responses)


def _state():
    plan = Plan(1, "Publish", (PlanStep("publish", "Publish", "Published"),))
    return AgentState("Publish", plan_history=(plan,))


def _receipt():
    return AIMessage(content=(
        'BEGIN STEP_COMPLETION\nCOMPLETED=true\nCOMPLETION_CRITERION_MET=true\n'
        'RESULT="Published"\nFILES_READ="README.md"\nFILES_MODIFIED="release.md"\n'
        'OBSERVATIONS="Publication confirmed"\nEND STEP_COMPLETION'
    ))


def test_executor_uses_no_tool_then_validates_separate_line_protocol_receipt(monkeypatch):
    async def load_tools(session): return []
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = Model(AIMessage(content="BEGIN NO_TOOL\nEND NO_TOOL"), _receipt())

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(_state(), 1, "publish")
    outcome = asyncio.run(run())
    assert outcome.execution.status == "completed"
    assert outcome.execution.result == "Published"
    assert outcome.context_update.files_read == ("README.md",)
    assert "NO_TOOL" in model.requests[0][0].content
    assert "STEP_COMPLETION" in model.requests[1][-1].content


def test_executor_rejects_step_completion_missing_its_required_receipt_proof(monkeypatch):
    async def load_tools(session): return []
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = Model(AIMessage(content="BEGIN NO_TOOL\nEND NO_TOOL"), AIMessage(content='BEGIN STEP_COMPLETION\nCOMPLETED=true\nRESULT="Published"\nEND STEP_COMPLETION'))

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(_state(), 1, "publish")
    assert asyncio.run(run()).execution.status == "failed"


def test_executor_executes_a_tool_call_with_accompanying_text(monkeypatch):
    def record() -> str:
        """Record the requested action."""
        return "recorded"

    async def load_tools(session):
        return [StructuredTool.from_function(record, name="record")]
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = Model(
        AIMessage(content="I'll record this.", tool_calls=[{"name": "record", "args": {}, "id": "1"}]),
        AIMessage(content="BEGIN NO_TOOL\nEND NO_TOOL"),
        _receipt(),
    )

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(_state(), 1, "publish")
    outcome = asyncio.run(run())
    assert outcome.execution.status == "completed", outcome.execution.error
