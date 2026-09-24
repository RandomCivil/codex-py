import asyncio

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from agent.executor import Executor
from memory.state import AgentState, Plan, PlanStep, StepContext


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


def _judge(completed: bool, evidence: str = "state proves artifact is available") -> AIMessage:
    if not completed:
        return AIMessage(content="BEGIN STEP_COMPLETION_PROGRESS\nALL_COMPLETED=false\nEND STEP_COMPLETION_PROGRESS")
    return AIMessage(content=(
        "BEGIN STEP_COMPLETION_PROGRESS\nALL_COMPLETED=true\n"
        "BEGIN COMPLETED_CRITERION\nNUMBER=1\n"
        f'EVIDENCE="{evidence}"\nEND COMPLETED_CRITERION\n'
        "END STEP_COMPLETION_PROGRESS"
    ))


def test_no_tool_step_judges_agent_state_and_step_context_before_handoff(monkeypatch):
    async def load_tools(_session):
        return []

    monkeypatch.setattr("agent.executor.MultiServerMCPClient", _Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = _Model(
        AIMessage(content="It is already done."),
        _judge(True),
        AIMessage(content="Published from existing context."),
    )

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(
                _state(),
                1,
                "publish",
                StepContext(observations=("The artifact is available in the workspace.",)),
            )

    outcome = asyncio.run(run())

    assert outcome.execution.status == "completed", outcome.execution.error
    assert outcome.execution.result == "Published from existing context."
    assert outcome.execution.completion_evidence == "state proves artifact is available"
    judge_request = model.requests[1]
    judge_payload = "\n".join(getattr(message, "content", "") for message in judge_request)
    assert "The artifact is available in the workspace." in judge_payload
    assert "Publish" in judge_payload
    assert model.tool_bindings[-1] == ()


def test_completed_step_repairs_invalid_tool_free_handoff_once(monkeypatch):
    async def load_tools(_session):
        return []

    monkeypatch.setattr("agent.executor.MultiServerMCPClient", _Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = _Model(
        AIMessage(content="Premature answer."),
        _judge(True),
        AIMessage(content="", tool_calls=[{"name": "unexpected", "args": {}, "id": "call-1"}]),
        AIMessage(content="Final handoff after repair."),
    )

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(_state(), 1, "publish")

    outcome = asyncio.run(run())

    assert outcome.execution.status == "completed"
    assert outcome.execution.result == "Final handoff after repair."
    assert len(model.requests) == 4
    assert all(binding == () for binding in model.tool_bindings[1:])


def test_pending_no_tool_response_continues_with_tools_and_consumes_round_budget(monkeypatch):
    async def load_tools(_session):
        def record():
            """Record the artifact as available."""
            return "artifact is available"

        return [StructuredTool.from_function(record, name="record")]

    monkeypatch.setattr("agent.executor.MultiServerMCPClient", _Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = _Model(
        AIMessage(content="Not finished yet."),
        _judge(False),
        AIMessage(content="", tool_calls=[{"name": "record", "args": {}, "id": "call-1"}]),
        _judge(True, "record confirms the artifact is available"),
        AIMessage(content="Completed after checking."),
    )

    async def run():
        async with Executor(model=model, max_rounds=2) as executor:
            return await executor.execute(_state(), 1, "publish")

    outcome = asyncio.run(run())

    assert outcome.execution.status == "completed", outcome.execution.error
    assert outcome.execution.result == "Completed after checking."
    assert outcome.execution.completion_evidence == "record confirms the artifact is available"
    assert "still pending" in model.requests[2][-1].content


def test_repeated_premature_no_tool_responses_fail_at_operational_budget(monkeypatch):
    async def load_tools(_session):
        return []

    monkeypatch.setattr("agent.executor.MultiServerMCPClient", _Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = _Model(
        AIMessage(content="Premature one."),
        _judge(False),
        AIMessage(content="Premature two."),
        _judge(False),
    )

    async def run():
        async with Executor(model=model, max_rounds=2) as executor:
            return await executor.execute(_state(), 1, "publish")

    outcome = asyncio.run(run())

    assert outcome.execution.status == "failed"
    assert outcome.execution.error == "tool round budget exhausted"
