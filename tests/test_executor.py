import asyncio
import json

import pytest

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from agent import Executor, PersistenceError
from memory.state import AgentState, Plan, PlanStep


class ControlledModel:
    def __init__(self):
        self.calls = []
        self.bindings = []

    def bind_tools(self, tools, **kwargs):
        self.tools = tools
        self.bind_options = kwargs
        self.bindings.append((tools, kwargs))
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

    assert execution.revision == 1
    assert execution.step_id == "publish"
    assert execution.status == "completed", execution.error
    assert execution.result == "Published the release. Completion criterion met: Release is available"
    assert state.step_executions == ()
    assert state.plan_history == (plan,)
    request = json.loads(model.calls[0][1].content)
    assert request["goal"] == "Prepare release"
    assert request["selected_plan_step"]["completion_criterion"] == "Release is available"
    assert request["plan_history"][0]["steps"][0]["id"] == "publish"
    assert request["memory_summary"] is None
    # MCP tools remain non-strict, while the separate completion tool is
    # forced and strictly typed.
    assert "response_format" not in model.bind_options
    completion_tools, completion_options = model.bindings[1]
    assert [tool.name for tool in completion_tools] == ["report_step_completion"]
    assert completion_options == {
        "tool_choice": "report_step_completion",
        "strict": True,
        "parallel_tool_calls": False,
    }


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

    assert execution.status == "completed"
    checkpoints = list(saver.list({"configurable": {"thread_id": "run-1:r1:s-publish"}}))
    assert any(
        checkpoint.checkpoint["channel_values"].get("execution", {}).get("status") == "running"
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

    assert execution.status == "failed"
    assert "step" in execution.error.lower()
    assert not opened


class ToolThenToolModel:
    def __init__(self):
        self.calls = []

    def bind_tools(self, tools, **kwargs):
        self.tools = tools
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

    assert execution.status == "failed"
    assert "round" in execution.error.lower()
    assert len(model.calls) == 2


def _completion_message(
    result="Published the release. Completion criterion met: Release is available",
    *,
    criterion_met=True,
):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "report_step_completion",
                "args": {
                    "completed": True,
                    "completion_criterion_met": criterion_met,
                    "result": result,
                },
                "id": "completion-call",
            }
        ],
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

    assert execution.status == "completed", execution.error
    assert len(model.calls) == 4
    errors = [message.content for message in model.calls[1] if isinstance(message, ToolMessage)]
    assert any("mcp unavailable" in str(error).lower() for error in errors)


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

    assert execution.status == "failed"
    assert "valid completion" in execution.error.lower()


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

    assert execution.status == "failed"
    assert "valid completion" in execution.error.lower()


def test_executor_accepts_a_strict_completion_tool_call(monkeypatch):
    model = FinalModel("Published the release. Completion criterion met: Release is available")
    tool = StructuredTool.from_function(record)
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool))
    _, state = _plan_and_state()

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(state, 1, "publish")

    execution = asyncio.run(run())

    assert execution.status == "completed", execution.error


def test_executor_accepts_a_structured_criterion_assertion_without_a_text_suffix(monkeypatch):
    class StructuredCriterionModel(FinalModel):
        async def ainvoke(self, messages):
            message = await super().ainvoke(messages)
            if len(self.calls) == 3:
                call = message.tool_calls[0]
                return message.model_copy(
                    update={
                        "tool_calls": [
                            {
                                **call,
                                "args": {
                                    **call["args"],
                                    "completion_criterion_met": True,
                                },
                            }
                        ]
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

    assert execution.status == "completed", execution.error
    assert execution.result == "功能清单已建立；下一步可扫描代码实现状态。"


class RecoverFromInvalidArgumentsModel:
    def __init__(self):
        self.calls = []

    def bind_tools(self, tools, **kwargs):
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

    assert execution.status == "completed", execution.error
    assert len(model.calls) == 4
    errors = [message.content for message in model.calls[1] if isinstance(message, ToolMessage)]
    assert any("value" in str(error).lower() for error in errors)


def test_executor_returns_nonzero_atom_exec_result_to_the_model(monkeypatch):
    class RecoverFromExecErrorModel:
        def __init__(self):
            self.calls = []

        def bind_tools(self, tools, **kwargs):
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

    assert execution.status == "completed", execution.error
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

    assert first.status == second.status == "completed"
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

    assert execution.status == "completed"
    assert received == [("done", "/workspace/project")]


def test_checkpointed_executor_fails_closed_before_tool_work(monkeypatch):
    _, state = _plan_and_state()
    executor = Executor(model=ControlledModel(), checkpointer=InMemorySaver(), run_id="run-1")

    class UnavailableGraph:
        async def ainvoke(self, payload, config):
            raise OSError("checkpoint storage is unavailable")

    async def install_unavailable_graph():
        executor._graph = UnavailableGraph()

    monkeypatch.setattr(executor, "_ensure_tools", install_unavailable_graph)

    async def run():
        async with executor:
            with pytest.raises(PersistenceError, match="stopped before further tool work"):
                await executor.execute(state, 1, "publish")

    asyncio.run(run())
