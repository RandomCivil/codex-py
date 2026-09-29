import asyncio

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


def _judge(completed: bool) -> AIMessage:
    if not completed:
        return AIMessage(content="BEGIN STEP_COMPLETION_PROGRESS\nALL_COMPLETED=false\nEND STEP_COMPLETION_PROGRESS")
    return AIMessage(content=(
        "BEGIN STEP_COMPLETION_PROGRESS\nALL_COMPLETED=true\n"
        "BEGIN COMPLETED_CRITERION\nNUMBER=1\n"
        'EVIDENCE="The artifact is available."\nEND COMPLETED_CRITERION\n'
        "END STEP_COMPLETION_PROGRESS"
    ))


def test_terminal_candidate_becomes_step_output_without_handoff(monkeypatch):
    async def load_tools(_session):
        return []

    monkeypatch.setattr("agent.executor.MultiServerMCPClient", _Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = _Model(AIMessage(content="Published the artifact."), _judge(True))

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(_state(), 1, "publish")

    outcome = asyncio.run(run())

    assert outcome.execution.status == "completed", outcome.execution.error
    assert outcome.execution.result == "Published the artifact."
    assert outcome.execution.completion_evidence == "The artifact is available."
    assert len(model.requests) == 2


def test_successful_tool_batch_waits_for_terminal_candidate_before_judging(monkeypatch):
    async def load_tools(_session):
        def publish():
            """Publish the artifact."""
            return "artifact is available"

        return [StructuredTool.from_function(publish, name="publish")]

    monkeypatch.setattr("agent.executor.MultiServerMCPClient", _Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = _Model(
        AIMessage(content="", tool_calls=[{"name": "publish", "args": {}, "id": "publish-1"}]),
        AIMessage(content="Published the artifact."),
        _judge(True),
    )

    async def run():
        async with Executor(model=model, max_rounds=2) as executor:
            return await executor.execute(_state(), 1, "publish")

    outcome = asyncio.run(run())

    assert outcome.execution.status == "completed", outcome.execution.error
    assert outcome.execution.result == "Published the artifact."
    assert len(model.requests) == 3
    assert model.requests[1][0].content == Executor._INSTRUCTIONS


def test_empty_terminal_response_continues_without_judging(monkeypatch):
    async def load_tools(_session):
        return []

    monkeypatch.setattr("agent.executor.MultiServerMCPClient", _Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = _Model(AIMessage(content=""), AIMessage(content="Published."), _judge(True))

    async def run():
        async with Executor(model=model, max_rounds=2) as executor:
            return await executor.execute(_state(), 1, "publish")

    outcome = asyncio.run(run())

    assert outcome.execution.status == "completed", outcome.execution.error
    assert len(model.requests) == 3
    assert "empty" in model.requests[1][-1].content


def test_negative_judgment_returns_candidate_to_operational_loop(monkeypatch):
    async def load_tools(_session):
        return []

    monkeypatch.setattr("agent.executor.MultiServerMCPClient", _Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = _Model(
        AIMessage(content="Premature answer."),
        _judge(False),
        AIMessage(content="Published after verification."),
        _judge(True),
    )

    async def run():
        async with Executor(model=model, max_rounds=2) as executor:
            return await executor.execute(_state(), 1, "publish")

    outcome = asyncio.run(run())

    assert outcome.execution.status == "completed", outcome.execution.error
    assert outcome.execution.result == "Published after verification."
    assert "still pending" in model.requests[2][-1].content
