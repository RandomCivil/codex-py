import asyncio
import json

import pytest

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from agent import Executor, PersistenceError
from agent.executor import _durable_context_payload
from agent.runtime_context import RuntimeContext
from memory.state import AgentState, ContextUpdate, ExecutionOutcome, Plan, PlanStep, StepContext


class ControlledModel:
    def __init__(self):
        self.calls = []
        self.bindings = []

    def bind_tools(self, tools, **kwargs):
        self.tools = tools
        self.bind_options = kwargs
        self.bindings.append((tools, kwargs))
        return self

    def bind(self, **kwargs):
        self.bind_options = kwargs
        self.bindings.append((None, kwargs))
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return AIMessage(
                content="",
                tool_calls=[{"name": "record", "args": {"value": "done"}, "id": "call-1"}],
            )
        if len(self.calls) == 2:
            return AIMessage(content="MCP work is complete.")
        return _completion_message()


class ChatOpenAIToolBindingModel(ControlledModel):
    """Exercise ChatOpenAI's real tool-schema conversion without network I/O."""

    def __init__(self):
        super().__init__()
        self._validator = ChatOpenAI(
            base_url="http://127.0.0.1:1/v1",
            api_key="test-key",
            model="test-model",
        )

    def bind_tools(self, tools, **kwargs):
        self._validator.bind_tools(tools, **kwargs)
        return super().bind_tools(tools, **kwargs)


class ControlledClient:
    def __init__(self):
        self.closed = False

    class _Session:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *args):
            return None

    def session(self, name):
        return self._Session()


def record(value: str) -> str:
    """Record a value."""
    return value


def test_executor_completes_a_plan_step_after_a_successful_tool_round(monkeypatch):
    model = ControlledModel()
    client = ControlledClient()
    tool = StructuredTool.from_function(record)
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool))

    plan = Plan(1, "Prepare release", (PlanStep("publish", "Publish it", "Release is available"),))
    state = AgentState("Prepare release", plan_history=(plan,))

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(state, 1, "publish")

    execution = asyncio.run(run())

    assert execution.execution.revision == 1
    assert execution.execution.step_id == "publish"
    assert execution.execution.status == "completed", execution.execution.error
    assert execution.execution.result == "Published the release. Completion criterion met: Release is available"
    assert state.step_executions == ()
    assert state.plan_history == (plan,)
    request = json.loads(model.calls[0][1].content)
    assert request["goal"] == "Prepare release"
    assert request["selected_plan_step"]["completion_criterion"] == "Release is available"
    assert request["plan_history"][0]["steps"][0]["id"] == "publish"
    assert request["memory_summary"] is None
    assert model.calls[0][2].content == (
        "## Step context\n\n"
        "### Files read\n"
        "- (none)\n\n"
        "### Files modified\n"
        "- (none)\n\n"
        "### Observations\n"
        "- (none)"
    )
    completion_prompt = model.calls[2][-1].content
    assert "handoff and evidence summary" in completion_prompt
    assert "subsequent Plan steps will receive" in completion_prompt
    assert "message transcript" in completion_prompt
    assert "top-level JSON object" not in completion_prompt
    assert 'Do not wrap the receipt in `{"type":"json_object","result":...}`' not in completion_prompt
    assert "Return the completion receipt" in completion_prompt
    # MCP tools remain non-strict; the separate completion reply uses JSON Schema.
    completion_tools, completion_options = model.bindings[1]
    assert completion_tools is None
    response_format = completion_options["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "step_completion"
    assert response_format["json_schema"]["strict"] is True


def test_executor_tool_selection_is_compatible_with_chat_openai_tool_binding(monkeypatch):
    model = ChatOpenAIToolBindingModel()
    tool = StructuredTool.from_function(record)
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool))
    plan = Plan(1, "Prepare release", (PlanStep("publish", "Publish it", "Release is available"),))
    state = AgentState("Prepare release", plan_history=(plan,))

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(state, 1, "publish")

    execution = asyncio.run(run())

    assert execution.execution.status == "completed", execution.execution.error
    assert model.bindings[0][1] == {}


def test_executor_json_object_mode_keeps_local_completion_validation(monkeypatch):
    model = ControlledModel()
    tool = StructuredTool.from_function(record)
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool))
    plan = Plan(1, "Prepare release", (PlanStep("publish", "Publish it", "Release is available"),))
    state = AgentState("Prepare release", plan_history=(plan,))

    async def run():
        async with Executor(model=model, response_format="json_object") as executor:
            return await executor.execute(state, 1, "publish")

    execution = asyncio.run(run())

    assert execution.execution.status == "completed", execution.execution.error
    assert model.bindings[1][0] is None
    assert model.bindings[1][1] == {"response_format": {"type": "json_object"}}
    completion_prompt = model.calls[2][-1].content
    assert "Follow this example exactly in shape" in completion_prompt
    assert '"completed":true,"completion_criterion_met":true' in completion_prompt
    assert "Do not include any other fields, including step_id, revision" in completion_prompt


def test_executor_requires_explicit_provider_values_instead_of_model_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    monkeypatch.setenv("OPENAI_MODEL", "environment-model")

    with pytest.raises(ValueError, match="requires explicit"):
        Executor()

    observed = {}

    class Model:
        pass

    def build_model(**kwargs):
        observed.update(kwargs)
        return Model()

    monkeypatch.setattr("agent.executor.ChatOpenAI", build_model)
    executor = Executor(
        base_url="https://provider.test/v1",
        api_key="configured-key",
        model_name="configured-model",
    )

    assert executor._model.__class__ is Model
    assert observed == {
        "base_url": "https://provider.test/v1",
        "api_key": "configured-key",
        "model": "configured-model",
        "max_retries": 0,
    }


def test_executor_runs_one_model_response_tool_call_batch_concurrently_and_waits(monkeypatch):
    class BatchModel:
        def __init__(self):
            self.calls = []
            self.completed_tools = []

        def bind_tools(self, tools, **kwargs):
            return self

        def bind(self, **kwargs):
            return self

        async def ainvoke(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "first", "args": {}, "id": "first-call"},
                        {"name": "second", "args": {}, "id": "second-call"},
                    ],
                )
            if len(self.calls) == 2:
                assert sorted(self.completed_tools) == ["first", "second"]
                results = [message for message in messages if isinstance(message, ToolMessage)]
                assert [(message.tool_call_id, message.content) for message in results] == [
                    ("first-call", "first"),
                    ("second-call", "second"),
                ]
                return AIMessage(content="MCP work is complete.")
            return _completion_message()

    model = BatchModel()
    both_started = asyncio.Event()
    release = asyncio.Event()
    started = []

    async def invoke(name):
        started.append(name)
        if len(started) == 2:
            both_started.set()
            release.set()
        await both_started.wait()
        await release.wait()
        model.completed_tools.append(name)
        return name

    async def first() -> str:
        """Run the first independent operation."""
        return await invoke("first")

    async def second() -> str:
        """Run the second independent operation."""
        return await invoke("second")

    tools = (
        StructuredTool.from_function(coroutine=first, name="first"),
        StructuredTool.from_function(coroutine=second, name="second"),
    )
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(*tools))
    _, state = _plan_and_state()

    async def run():
        async with Executor(model=model) as executor:
            return await asyncio.wait_for(executor.execute(state, 1, "publish"), timeout=1)

    execution = asyncio.run(run())

    assert execution.execution.status == "completed", execution.execution.error
    assert started == ["first", "second"]


def test_executor_returns_context_update_and_sends_step_context_separately(monkeypatch):
    class ContextModel(ControlledModel):
        async def ainvoke(self, messages):
            if len(self.calls) == 0:
                assert messages[1].content.startswith("{")
                assert messages[2].content == (
                    "## Step context\n\n"
                    "### Files read\n"
                    "- docs/overview.md\n\n"
                    "### Files modified\n"
                    "- src/agent.py\n\n"
                    "### Observations\n"
                    "- The execution boundary is already isolated."
                )
            if len(self.calls) == 2:
                self.calls.append(messages)
                return AIMessage(
                    content=json.dumps(
                        {
                            "completed": True,
                            "completion_criterion_met": True,
                            "result": "Published the release.",
                            "files_read": ["docs/overview.md"],
                            "files_modified": ["src/agent.py"],
                            "observations": ["The execution boundary is already isolated."],
                        }
                    ),
                )
            return await super().ainvoke(messages)

    model = ContextModel()
    tool = StructuredTool.from_function(record)
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool))
    _, state = _plan_and_state()

    async def run():
        async with Executor(model=model, tool_cwd="/workspace/project") as executor:
            return await executor.execute(
                state,
                1,
                "publish",
                StepContext(
                    files_read=("docs/overview.md",),
                    files_modified=("src/agent.py",),
                    observations=("The execution boundary is already isolated.",),
                ),
            )

    outcome = asyncio.run(run())

    assert isinstance(outcome, ExecutionOutcome)
    assert outcome.execution.status == "completed", outcome.execution.error
    assert outcome.context_update == ContextUpdate(
        files_read=("docs/overview.md",),
        files_modified=("src/agent.py",),
        observations=("The execution boundary is already isolated.",),
    )


def test_executor_completes_a_plan_step_without_an_mcp_tool_call(monkeypatch):
    class NoToolCompletionModel:
        def __init__(self):
            self.calls = []

        def bind_tools(self, tools, **kwargs):
            return self

        def bind(self, **kwargs):
            return self

        async def ainvoke(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return AIMessage(content="The selected Plan step is already complete.")
            return _completion_message("The selected Plan step was already complete; Release is available.")

    model = NoToolCompletionModel()
    tool = StructuredTool.from_function(record)
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool))
    _, state = _plan_and_state()

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(state, 1, "publish")

    execution = asyncio.run(run())

    assert execution.execution.status == "completed", execution.execution.error
    assert execution.execution.result == "The selected Plan step was already complete; Release is available."
    assert len(model.calls) == 2
    assert not any(isinstance(message, ToolMessage) for message in model.calls[1])


def test_executor_checkpoints_running_before_model_work(monkeypatch):
    model = ControlledModel()
    tool = StructuredTool.from_function(record)
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool))
    saver = InMemorySaver()
    plan = Plan(1, "Prepare release", (PlanStep("publish", "Publish it", "Release is available"),))
    state = AgentState("Prepare release", plan_history=(plan,))

    async def run():
        async with Executor(
            model=model,
            checkpointer=saver,
            thread_id="run-1:r1:s-publish",
        ) as executor:
            return await executor.execute(state, 1, "publish")

    execution = asyncio.run(run())

    assert execution.execution.status == "completed"
    checkpoints = list(saver.list({"configurable": {"thread_id": "run-1:r1:s-publish"}}))
    assert any(
        checkpoint.checkpoint["channel_values"].get("execution") is not None
        and checkpoint.checkpoint["channel_values"]["execution"].status == "running"
        for checkpoint in checkpoints
    )


def test_executor_recovery_uses_a_fresh_attempt_thread_and_reconciliation_marker(monkeypatch):
    model = ControlledModel()
    tool = StructuredTool.from_function(record)
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool))
    saver = InMemorySaver()
    plan = Plan(1, "Prepare release", (PlanStep("publish", "Publish it", "Release is available"),))
    state = AgentState("Prepare release", plan_history=(plan,))

    async def run():
        async with Executor(model=model, checkpointer=saver, run_id="run-1") as executor:
            return await executor.execute(state, 1, "publish", recovery=True, attempt=2)

    outcome = asyncio.run(run())

    assert outcome.execution.status == "completed"
    assert "fresh execution attempt" in model.calls[0][1].content
    checkpoints = list(saver.list({"configurable": {"thread_id": "run-1:r1:spublish:a2"}}))
    assert checkpoints


def test_executor_checkpoint_keeps_running_marker_but_not_transient_tool_messages(monkeypatch):
    model = ControlledModel()
    tool = StructuredTool.from_function(record)
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool))
    saver = InMemorySaver()
    _, state = _plan_and_state()

    async def run():
        async with Executor(model=model, checkpointer=saver, run_id="run-ctx") as executor:
            return await executor.execute(state, 1, "publish")

    outcome = asyncio.run(run())
    assert outcome.execution.status == "completed"
    checkpoints = list(saver.list({"configurable": {"thread_id": "run-ctx:r1:spublish"}}))
    assert checkpoints
    assert all("messages" not in checkpoint.checkpoint["channel_values"] for checkpoint in checkpoints)
    assert any(
        checkpoint.checkpoint["channel_values"].get("execution").status == "running"
        for checkpoint in checkpoints
    )


def _load_tools(*tools):
    async def load(session, **kwargs):
        return list(tools)

    return load


def test_executor_rejects_an_unknown_plan_step_without_opening_mcp(monkeypatch):
    opened = False

    def client(config):
        nonlocal opened
        opened = True
        raise AssertionError("MCP must not open for an invalid reference")

    monkeypatch.setattr("agent.executor.MultiServerMCPClient", client)
    plan = Plan(1, "Prepare release", (PlanStep("publish", "Publish it", "Release is available"),))

    async def run():
        async with Executor(model=ControlledModel()) as executor:
            return await executor.execute(AgentState("Prepare release", (plan,)), 1, "missing")

    execution = asyncio.run(run())

    assert execution.execution.status == "failed"
    assert "step" in execution.execution.error.lower()
    assert not opened


class ToolThenToolModel:
    def __init__(self):
        self.calls = []

    def bind_tools(self, tools, **kwargs):
        self.tools = tools
        return self

    def bind(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(
            content="",
            tool_calls=[{"name": "record", "args": {"value": "again"}, "id": f"call-{len(self.calls)}"}],
        )


def _plan_and_state():
    plan = Plan(1, "Prepare release", (PlanStep("publish", "Publish it", "Release is available"),))
    return plan, AgentState("Prepare release", plan_history=(plan,))


def test_executor_runtime_durable_state_is_structured_and_json_serializable():
    plan, state = _plan_and_state()
    payload = _durable_context_payload(
        state,
        1,
        plan.steps[0],
        StepContext(files_read=["README.md"], observations=["release inspected"]),
    )

    assert payload["agent_state"]["goal"] == "Prepare release"
    assert payload["step_context"]["files_read"] == ["README.md"]
    assert json.loads(json.dumps(payload)) == payload


def test_executor_fails_when_tool_round_budget_is_exhausted(monkeypatch):
    model = ToolThenToolModel()
    tool = StructuredTool.from_function(record)
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool))
    _, state = _plan_and_state()

    async def run():
        async with Executor(model=model, max_rounds=1) as executor:
            return await executor.execute(state, 1, "publish")

    execution = asyncio.run(run())

    assert execution.execution.status == "failed"
    assert "round" in execution.execution.error.lower()
    assert len(model.calls) == 2


def _completion_message(
    result="Published the release. Completion criterion met: Release is available",
    *,
    criterion_met=True,
):
    return AIMessage(
        content=json.dumps(
            {
                "completed": True,
                "completion_criterion_met": criterion_met,
                "result": result,
                "files_read": [],
                "files_modified": [],
                "observations": [],
            }
        )
    )


class FinalModel:
    def __init__(self, completion_result, tool_name="record", tool_args=None, criterion_met=True):
        self.completion_result = completion_result
        self.tool_name = tool_name
        self.tool_args = tool_args or {"value": "done"}
        self.criterion_met = criterion_met
        self.calls = []

    def bind_tools(self, tools, **kwargs):
        return self

    def bind(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return AIMessage(
                content="",
                tool_calls=[{"name": self.tool_name, "args": self.tool_args, "id": "call-1"}],
            )
        if len(self.calls) == 2:
            return AIMessage(content="MCP work is complete.")
        return _completion_message(self.completion_result, criterion_met=self.criterion_met)


def test_executor_binds_the_effective_mcp_tool_set_in_canonical_name_order(monkeypatch):
    class CapturingFinalModel(FinalModel):
        def bind_tools(self, tools, **kwargs):
            self.tools = tools
            return super().bind_tools(tools, **kwargs)

    executed = []

    def alpha(value: str) -> str:
        """Record that the allowed tool executed."""
        executed.append(value)
        return value

    model = CapturingFinalModel(
        "Published the release. Completion criterion met: Release is available",
        tool_name="alpha",
    )
    zebra = StructuredTool.from_function(record, name="zebra")
    outside_allowlist = StructuredTool.from_function(record, name="outside_allowlist")
    alpha_tool = StructuredTool.from_function(alpha, name="alpha")
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr(
        "agent.executor.load_mcp_tools", _load_tools(zebra, outside_allowlist, alpha_tool)
    )
    _, state = _plan_and_state()

    async def run():
        async with Executor(model=model, tool_allowlist=("zebra", "alpha")) as executor:
            return await executor.execute(state, 1, "publish")

    execution = asyncio.run(run())

    assert execution.execution.status == "completed", execution.execution.error
    assert execution.execution.result == "Published the release. Completion criterion met: Release is available"
    assert [tool.name for tool in model.tools] == ["alpha", "zebra"]
    assert executed == ["done"]


def test_executor_returns_an_mcp_tool_error_to_the_model(monkeypatch):
    class RecoverFromMcpErrorModel(RecoverFromInvalidArgumentsModel):
        async def ainvoke(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 2:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "recover", "args": {"value": "done"}, "id": "recover-call"}],
                )
            if len(self.calls) == 3:
                return AIMessage(content="MCP work is complete.")
            if len(self.calls) == 4:
                return _completion_message()
            return AIMessage(
                content="",
                tool_calls=[{"name": "record", "args": {"value": "done"}, "id": "broken-call"}],
            )

    model = RecoverFromMcpErrorModel()

    def broken(value: str) -> str:
        """Raise an MCP failure."""
        raise RuntimeError("MCP unavailable")

    tool = StructuredTool.from_function(broken, name="record")
    recovery_tool = StructuredTool.from_function(record, name="recover")
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool, recovery_tool))
    _, state = _plan_and_state()

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(state, 1, "publish")

    execution = asyncio.run(run())

    assert execution.execution.status == "completed", execution.execution.error
    assert len(model.calls) == 4
    errors = [message.content for message in model.calls[1] if isinstance(message, ToolMessage)]
    assert any("mcp unavailable" in str(error).lower() for error in errors)


def test_executor_returns_a_toolnode_exception_to_the_model(monkeypatch):
    class RecoverFromToolNodeExceptionModel(FinalModel):
        async def ainvoke(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "record", "args": {"value": "first"}, "id": "first-call"}],
                )
            if len(self.calls) == 2:
                errors = [message.content for message in messages if isinstance(message, ToolMessage)]
                assert errors == ["MCP tool failed: connection lost"]
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "record", "args": {"value": "recovered"}, "id": "recovery-call"}],
                )
            if len(self.calls) == 3:
                return AIMessage(content="MCP work is complete.")
            return _completion_message("Recovered after the MCP connection error.")

    class FailingThenSuccessfulToolNode:
        calls = 0

        def __init__(self, *args, **kwargs):
            pass

        async def ainvoke(self, state):
            type(self).calls += 1
            if type(self).calls == 1:
                raise OSError("connection lost")
            call = state["messages"][-1].tool_calls[0]
            return {
                "messages": [
                    ToolMessage(
                        content="recovered",
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                ]
            }

    model = RecoverFromToolNodeExceptionModel("unused")
    tool = StructuredTool.from_function(record)
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool))
    monkeypatch.setattr("agent.executor.ToolNode", FailingThenSuccessfulToolNode)
    _, state = _plan_and_state()

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(state, 1, "publish")

    execution = asyncio.run(run())

    assert execution.execution.status == "completed", execution.execution.error
    assert len(model.calls) == 4


def test_executor_rejects_a_completion_result_without_criterion_evidence(monkeypatch):
    model = FinalModel("done", criterion_met=False)
    tool = StructuredTool.from_function(record)
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool))
    _, state = _plan_and_state()

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(state, 1, "publish")

    execution = asyncio.run(run())

    assert execution.execution.status == "failed"
    assert "valid completion" in execution.execution.error.lower()


@pytest.mark.parametrize("unsafe_path", ["/workspace/project/file.py", "../file.py", "src/../file.py"])
def test_executor_rejects_an_unsafe_context_path(monkeypatch, unsafe_path):
    class UnsafeContextModel(FinalModel):
        async def ainvoke(self, messages):
            message = await super().ainvoke(messages)
            if len(self.calls) == 3:
                return message.model_copy(
                    update={
                        "content": json.dumps(
                            {**json.loads(message.content), "files_read": [unsafe_path]}
                        )
                    }
                )
            return message

    model = UnsafeContextModel("The release is available.")
    tool = StructuredTool.from_function(record)
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool))
    _, state = _plan_and_state()

    async def run():
        async with Executor(model=model, tool_cwd="/workspace/project") as executor:
            return await executor.execute(state, 1, "publish")

    execution = asyncio.run(run())

    assert execution.execution.status == "failed"
    assert "valid completion" in execution.execution.error.lower()


def test_executor_rejects_prose_completion_after_mcp_work(monkeypatch):
    class ProseCompletionModel(FinalModel):
        async def ainvoke(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": self.tool_name, "args": self.tool_args, "id": "call-1"}],
                )
            return AIMessage(content='completed: true, result: "The work is done."')

    model = ProseCompletionModel("unused")
    tool = StructuredTool.from_function(record)
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool))
    _, state = _plan_and_state()

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(state, 1, "publish")

    execution = asyncio.run(run())

    assert execution.execution.status == "failed"
    assert "valid completion" in execution.execution.error.lower()


def test_executor_accepts_a_strict_json_schema_completion(monkeypatch):
    model = FinalModel("Published the release. Completion criterion met: Release is available")
    tool = StructuredTool.from_function(record)
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool))
    _, state = _plan_and_state()

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(state, 1, "publish")

    execution = asyncio.run(run())

    assert execution.execution.status == "completed", execution.execution.error


def test_executor_accepts_a_structured_criterion_assertion_without_a_text_suffix(monkeypatch):
    class StructuredCriterionModel(FinalModel):
        async def ainvoke(self, messages):
            message = await super().ainvoke(messages)
            if len(self.calls) == 3:
                return message.model_copy(
                    update={
                        "content": json.dumps(
                            {**json.loads(message.content), "completion_criterion_met": True}
                        )
                    }
                )
            return message

    model = StructuredCriterionModel("功能清单已建立；下一步可扫描代码实现状态。")
    tool = StructuredTool.from_function(record)
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool))
    _, state = _plan_and_state()

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(state, 1, "publish")

    execution = asyncio.run(run())

    assert execution.execution.status == "completed", execution.execution.error
    assert execution.execution.result == "功能清单已建立；下一步可扫描代码实现状态。"


class RecoverFromInvalidArgumentsModel:
    def __init__(self):
        self.calls = []

    def bind_tools(self, tools, **kwargs):
        return self

    def bind(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if len(self.calls) == 2:
            return AIMessage(
                content="",
                tool_calls=[{"name": "record", "args": {"value": "done"}, "id": "record-call"}],
            )
        if len(self.calls) == 3:
            return AIMessage(content="MCP work is complete.")
        if len(self.calls) == 4:
            return _completion_message()
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "record", "args": {}, "id": "invalid-call"},
            ],
        )


def test_executor_returns_invalid_tool_arguments_to_the_model(monkeypatch):
    model = RecoverFromInvalidArgumentsModel()
    tool = StructuredTool.from_function(record)
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool))
    _, state = _plan_and_state()

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(state, 1, "publish")

    execution = asyncio.run(run())

    assert execution.execution.status == "completed", execution.execution.error
    assert len(model.calls) == 4
    errors = [message.content for message in model.calls[1] if isinstance(message, ToolMessage)]
    assert any("value" in str(error).lower() for error in errors)


def test_executor_returns_nonzero_atom_exec_result_to_the_model(monkeypatch):
    class RecoverFromExecErrorModel:
        def __init__(self):
            self.calls = []

        def bind_tools(self, tools, **kwargs):
            return self

        def bind(self, **kwargs):
            return self

        async def ainvoke(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "exec", "args": {"command": "false"}, "id": "exec-call"}],
                )
            if len(self.calls) == 2:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "record", "args": {"value": "done"}, "id": "record-call"}],
                )
            if len(self.calls) == 3:
                return AIMessage(content="MCP work is complete.")
            return _completion_message()

    model = RecoverFromExecErrorModel()

    def atom_exec(command: str) -> str:
        """Return a failed Atom command result."""
        return json.dumps({"exit_code": 7, "stdout": "", "stderr": "failed"})

    tool = StructuredTool.from_function(atom_exec, name="exec")
    record_tool = StructuredTool.from_function(record)
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool, record_tool))
    _, state = _plan_and_state()

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(state, 1, "publish")

    execution = asyncio.run(run())

    assert execution.execution.status == "completed", execution.execution.error
    assert len(model.calls) == 4
    results = [message.content for message in model.calls[1] if isinstance(message, ToolMessage)]
    assert any("failed" in str(result).lower() for result in results)


def test_executor_rejects_a_non_positive_round_budget():
    with pytest.raises(ValueError, match="round"):
        Executor(model=ControlledModel(), max_rounds=0)


def test_executor_forwards_host_configuration_and_reuses_one_mcp_session(monkeypatch):
    configurations = []
    sessions = []
    load_count = 0
    tool = StructuredTool.from_function(record)

    class Session:
        async def __aenter__(self):
            sessions.append(self)
            return object()

        async def __aexit__(self, *args):
            self.closed = True

    class Client:
        def __init__(self, config):
            configurations.append(config)

        def session(self, name):
            return Session()

    async def load(session, **kwargs):
        nonlocal load_count
        load_count += 1
        return [tool]

    monkeypatch.setenv("INHERITED_EXECUTOR_VALUE", "inherited")
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load)
    _, state = _plan_and_state()
    model = ControlledModel()

    async def run():
        async with Executor(
            model=model,
            command="custom-atom",
            args=("--stdio",),
            cwd="/tmp/atom",
            env={"SUPPLEMENTAL_EXECUTOR_VALUE": "provided"},
            tool_allowlist=("record",),
        ) as executor:
            first = await executor.execute(state, 1, "publish")
            model.calls.clear()
            second = await executor.execute(state, 1, "publish")
            return first, second

    first, second = asyncio.run(run())

    assert first.execution.status == second.execution.status == "completed"
    assert len(configurations) == 1
    assert configurations[0]["atom"]["transport"] == "stdio"
    assert configurations[0]["atom"]["command"] == "custom-atom"
    assert configurations[0]["atom"]["args"] == ["--stdio"]
    assert configurations[0]["atom"]["cwd"] == "/tmp/atom"
    assert configurations[0]["atom"]["env"]["INHERITED_EXECUTOR_VALUE"] == "inherited"
    assert configurations[0]["atom"]["env"]["SUPPLEMENTAL_EXECUTOR_VALUE"] == "provided"
    assert load_count == 1
    assert len(sessions) == 1
    assert sessions[0].closed


def test_executor_forces_the_agent_working_directory_into_atom_tool_calls(monkeypatch):
    model = FinalModel(
        "Published the release. Completion criterion met: Release is available",
        tool_args={"value": "done", "cwd": "/model-selected-directory"},
    )
    received = []

    def atom_tool(value: str, cwd: str) -> str:
        """Record the supplied working directory."""
        received.append((value, cwd))
        return value

    tool = StructuredTool.from_function(atom_tool, name="record")
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool))
    _, state = _plan_and_state()

    async def run():
        async with Executor(model=model, tool_cwd="/workspace/project") as executor:
            return await executor.execute(state, 1, "publish")

    execution = asyncio.run(run())

    assert execution.execution.status == "completed"
    assert received == [("done", "/workspace/project")]


def test_checkpointed_executor_fails_closed_before_tool_work(monkeypatch):
    _, state = _plan_and_state()
    executor = Executor(model=ControlledModel(), checkpointer=InMemorySaver(), run_id="run-1")

    class UnavailableGraph:
        async def ainvoke(self, payload, config):
            raise OSError("checkpoint storage is unavailable")

    async def install_unavailable_graph(state, revision, step):
        executor._graph = UnavailableGraph()
        return await executor._graph.ainvoke({}, None)

    monkeypatch.setattr(executor, "_execute_with_tools", install_unavailable_graph)

    async def run():
        async with executor:
            with pytest.raises(PersistenceError, match="stopped before further tool work"):
                await executor.execute(state, 1, "publish")

    asyncio.run(run())
