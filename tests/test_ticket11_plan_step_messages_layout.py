import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from agent.executor import Executor
from agent.runtime_context import RuntimeContextPolicy
from memory.state import AgentState, Plan, PlanStep


def test_executor_validates_runtime_context_layout():
    with pytest.raises(ValueError, match="runtime context layout"):
        Executor(model=object(), runtime_context_layout="invalid")


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


class _TrackingPolicy(RuntimeContextPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maintain_calls = []

    async def maintain(self, durable_state, **kwargs):
        self.maintain_calls.append(kwargs)
        return await super().maintain(durable_state, **kwargs)

    def fresh_for_invocation(self):
        return self


def _state():
    plan = Plan(1, "Publish", (PlanStep("publish", "Publish", "The artifact is available"),))
    return AgentState("Publish", plan_history=(plan,))


def _judge(completed=False):
    if not completed:
        return AIMessage(content="BEGIN STEP_COMPLETION_PROGRESS\nALL_COMPLETED=false\nEND STEP_COMPLETION_PROGRESS")
    return AIMessage(content=(
        "BEGIN STEP_COMPLETION_PROGRESS\nALL_COMPLETED=true\n"
        "BEGIN COMPLETED_CRITERION\nNUMBER=1\nEVIDENCE=\"artifact is available\"\n"
        "END COMPLETED_CRITERION\nEND STEP_COMPLETION_PROGRESS"
    ))


def test_plan_step_messages_layout_orders_operational_context_and_retains_failed_correction(monkeypatch):
    async def load_tools(_session):
        def succeed():
            """Produce the artifact."""
            return "artifact is available"

        def fail():
            """Fail the operation."""
            raise RuntimeError("permission denied")

        return [
            StructuredTool.from_function(succeed, name="succeed"),
            StructuredTool.from_function(fail, name="fail"),
        ]

    monkeypatch.setattr("agent.executor.MultiServerMCPClient", _Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = _Model(
        AIMessage(content="", tool_calls=[
            {"name": "succeed", "args": {}, "id": "ok-1"},
            {"name": "fail", "args": {}, "id": "bad-1"},
        ]),
        _judge(),
        AIMessage(content="Continue checking."),
        _judge(),
    )

    async def run():
        policy = RuntimeContextPolicy(None, layout="messages")
        async with Executor(model=model, context_policy=policy, max_rounds=2) as executor:
            return await executor.execute(_state(), 1, "publish")

    outcome = asyncio.run(run())
    assert outcome.execution.status == "failed"

    first = model.requests[0]
    assert [type(message) for message in first] == [
        SystemMessage, HumanMessage, HumanMessage, HumanMessage, HumanMessage, HumanMessage,
        HumanMessage,
    ]
    assert first[1].content.startswith('{"goal": "Publish"')
    assert "## Durable state" in first[2].content
    assert "## Observations" in first[3].content
    assert "## Goal" in first[-3].content
    assert "## Completion criteria" in first[-1].content

    correction = next(
        request for request in model.requests
        if any(isinstance(message, ToolMessage) and message.status == "error" for message in request)
    )
    assert any(
        "permission denied" in message.content
        for message in correction
        if isinstance(message, ToolMessage)
    )
    assert [message.tool_call_id for message in correction if isinstance(message, ToolMessage)] == ["ok-1", "bad-1"]
    assert correction.index(next(message for message in correction if isinstance(message, HumanMessage) and "## Goal" in message.content)) > correction.index(next(message for message in correction if isinstance(message, HumanMessage) and "## Observations" in message.content))


def test_plan_step_messages_layout_judge_reconstructs_only_successful_tool_pairs(monkeypatch):
    async def load_tools(_session):
        def succeed():
            """Produce the artifact."""
            return "artifact is available"

        def fail():
            """Fail the operation."""
            raise RuntimeError("permission denied")

        return [StructuredTool.from_function(succeed, name="succeed"), StructuredTool.from_function(fail, name="fail")]

    monkeypatch.setattr("agent.executor.MultiServerMCPClient", _Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = _Model(
        AIMessage(content="", tool_calls=[
            {"name": "succeed", "args": {}, "id": "ok-1"},
            {"name": "fail", "args": {}, "id": "bad-1"},
        ]),
        _judge(),
        AIMessage(content="No progress."),
    )

    policy = _TrackingPolicy(None, layout="messages")

    async def run():
        async with Executor(model=model, context_policy=policy, max_rounds=1) as executor:
            return await executor.execute(_state(), 1, "publish")

    asyncio.run(run())
    judge = model.requests[1]
    assert [message.tool_call_id for message in judge if isinstance(message, ToolMessage)] == ["ok-1"]
    assert all(getattr(message, "tool_call_id", None) != "bad-1" for message in judge)
    assert "permission denied" not in "\n".join(getattr(message, "content", "") for message in judge)
    assert isinstance(judge[0], SystemMessage)
    assert isinstance(judge[1], HumanMessage)
    assert judge.index(next(message for message in judge if isinstance(message, HumanMessage) and "## Goal" in message.content)) > judge.index(next(message for message in judge if isinstance(message, HumanMessage) and "## Observations" in message.content))
    judge_budget = next(
        kwargs for kwargs in policy.maintain_calls
        if kwargs.get("prefix_messages")
        and kwargs["prefix_messages"][0].content.startswith("You are a tool-free completion judge")
    )
    assert "selected_step" in judge_budget["feedback_messages"][0].content
    assert judge_budget["completion_criteria_status"].endswith("The artifact is available")
