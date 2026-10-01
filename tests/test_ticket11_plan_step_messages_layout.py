import asyncio
import json

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
        SystemMessage, HumanMessage, HumanMessage, HumanMessage, HumanMessage,
    ]
    assert first[1].content.startswith('{"goal": "Publish"')
    assert "## Durable state" in first[2].content
    assert json.loads(first[2].content.split("\n\n", 1)[1]) == {
        "step_context": {"files_read": [], "files_modified": [], "observations": []},
    }
    assert "## Observations" in first[3].content
    assert not any(message.content.startswith(("## Goal", "## Steps")) for message in first)
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
    assert correction[-1].content.startswith("## Completion criteria")
    assert not any(message.content.startswith(("## Goal", "## Steps")) for message in correction)


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
        AIMessage(content="No progress."),
        _judge(),
    )

    policy = _TrackingPolicy(None, layout="messages")

    async def run():
        async with Executor(model=model, context_policy=policy, max_rounds=2) as executor:
            return await executor.execute(_state(), 1, "publish")

    asyncio.run(run())
    judge = model.requests[2]
    assert [message.tool_call_id for message in judge if isinstance(message, ToolMessage)] == ["ok-1"]
    assert all(getattr(message, "tool_call_id", None) != "bad-1" for message in judge)
    assert "permission denied" not in "\n".join(getattr(message, "content", "") for message in judge)
    assert isinstance(judge[0], SystemMessage)
    assert isinstance(judge[1], HumanMessage)
    assert judge[-1].content.startswith("## Completion criteria")
    assert not any(message.content.startswith(("## Goal", "## Steps")) for message in judge)
    assert json.loads(judge[1].content)["goal"] == "Publish"
    judge_budget = next(
        kwargs for kwargs in policy.maintain_calls
        if kwargs.get("prefix_messages")
        and kwargs["prefix_messages"][0].content.startswith("You are a tool-free completion judge")
    )
    assert "selected_step" in judge_budget["feedback_messages"][0].content
    assert judge_budget["completion_criteria_status"].endswith("The artifact is available")


def test_plan_step_messages_layout_terminal_candidate_receives_settled_tool_result(monkeypatch):
    async def load_tools(_session):
        def succeed():
            """Produce the artifact."""
            return "artifact is available"

        return [StructuredTool.from_function(succeed, name="succeed")]

    monkeypatch.setattr("agent.executor.MultiServerMCPClient", _Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = _Model(
        AIMessage(content="", tool_calls=[{"name": "succeed", "args": {}, "id": "ok-1"}]),
        AIMessage(content="Published the artifact."),
        _judge(completed=True),
    )

    async def run():
        async with Executor(model=model, runtime_context_layout="messages") as executor:
            return await executor.execute(_state(), 1, "publish")

    outcome = asyncio.run(run())
    assert outcome.execution.status == "completed", outcome.execution.error
    candidate = model.requests[1]
    assert [message.tool_call_id for message in candidate if isinstance(message, ToolMessage)] == ["ok-1"]
    assert next(message for message in candidate if isinstance(message, ToolMessage)).content == "artifact is available"
    assert any(isinstance(message, AIMessage) and message.tool_calls[0]["id"] == "ok-1" for message in candidate)
    assert "[pending]" in candidate[-1].content


def test_plan_step_messages_layout_rejected_candidate_keeps_tool_result(monkeypatch):
    async def load_tools(_session):
        def succeed():
            """Produce the artifact."""
            return "artifact is available"

        return [StructuredTool.from_function(succeed, name="succeed")]

    monkeypatch.setattr("agent.executor.MultiServerMCPClient", _Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = _Model(
        AIMessage(content="", tool_calls=[{"name": "succeed", "args": {}, "id": "ok-1"}]),
        AIMessage(content="Premature."),
        _judge(completed=False),
        AIMessage(content="Published the artifact."),
        _judge(completed=True),
    )

    async def run():
        async with Executor(model=model, runtime_context_layout="messages", max_rounds=3) as executor:
            return await executor.execute(_state(), 1, "publish")

    outcome = asyncio.run(run())
    assert outcome.execution.status == "completed", outcome.execution.error
    retry = model.requests[3]
    assert [message.tool_call_id for message in retry if isinstance(message, ToolMessage)] == ["ok-1"]
    feedback_index = next(
        index for index, message in enumerate(retry)
        if isinstance(message, HumanMessage) and "proposed Step answer was not accepted" in message.content
    )
    criteria_index = next(
        index for index, message in enumerate(retry)
        if isinstance(message, HumanMessage) and message.content.startswith("## Completion criteria")
    )
    assert feedback_index < criteria_index


def test_plan_step_messages_judge_rejects_native_tool_calls(monkeypatch):
    invocations = []

    async def load_tools(_session):
        def succeed():
            """Produce the artifact."""
            invocations.append("succeed")
            return "artifact is available"

        return [StructuredTool.from_function(succeed, name="succeed")]

    class TrackingModel(_Model):
        def __init__(self, *responses):
            super().__init__(*responses)
            self.bindings = []

        def bind_tools(self, tools):
            self.bindings.append(tuple(tools))
            return self

    monkeypatch.setattr("agent.executor.MultiServerMCPClient", _Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    judge_with_call = _judge(completed=True).model_copy(update={
        "tool_calls": [{"name": "succeed", "args": {}, "id": "judge-call"}],
    })
    model = TrackingModel(
        AIMessage(content="", tool_calls=[{"name": "succeed", "args": {}, "id": "ok-1"}]),
        AIMessage(content="Published the artifact."),
        judge_with_call,
        _judge(completed=True),
    )

    async def run():
        async with Executor(model=model, runtime_context_layout="messages") as executor:
            return await executor.execute(_state(), 1, "publish")

    outcome = asyncio.run(run())
    assert outcome.execution.status == "completed", outcome.execution.error
    assert invocations == ["succeed"]
    assert len(model.requests) == 4
    assert model.bindings[1:] == [()]
    assert "Validation error:" in "\n".join(message.content for message in model.requests[3] if isinstance(message, HumanMessage))
    assert model.requests[3][-1].content.startswith("## Completion criteria")
