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

    def bind_tools(self, tools):
        self.tools = tools
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return AIMessage(
                content="",
                tool_calls=[{"name": "record", "args": {"value": "done"}, "id": "call-1"}],
            )
        return AIMessage(
            content=json.dumps(
                {
                    "completed": True,
                    "result": "Published the release. Completion criterion met: Release is available",
                }
            )
        )


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


def _load_tools(tool):
    async def load(session, **kwargs):
        return [tool]

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

    def bind_tools(self, tools):
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


class FinalModel:
    def __init__(self, content, tool_name="record", tool_args=None):
        self.content = content
        self.tool_name = tool_name
        self.tool_args = tool_args or {"value": "done"}
        self.calls = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return AIMessage(
                content="",
                tool_calls=[{"name": self.tool_name, "args": self.tool_args, "id": "call-1"}],
            )
        return AIMessage(content=self.content)


def test_executor_fails_immediately_when_an_mcp_tool_raises(monkeypatch):
    model = FinalModel(json.dumps({"completed": True, "result": "done"}))

    def broken(value: str) -> str:
        """Raise an MCP failure."""
        raise RuntimeError("MCP unavailable")

    tool = StructuredTool.from_function(broken, name="record")
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool))
    _, state = _plan_and_state()

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(state, 1, "publish")

    execution = asyncio.run(run())

    assert execution.status == "failed"
    assert "mcp unavailable" in execution.error.lower()
    assert len(model.calls) == 1


def test_executor_rejects_a_completion_result_without_criterion_evidence(monkeypatch):
    model = FinalModel(json.dumps({"completed": True, "result": "done"}))
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


class InvalidThenValidToolModel:
    def __init__(self):
        self.calls = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(
            content="",
            tool_calls=[
                {"name": "missing", "args": {}, "id": "missing-call"},
                {"name": "record", "args": {"value": "done"}, "id": "record-call"},
            ],
        )


def test_executor_does_not_return_to_the_model_after_any_tool_error(monkeypatch):
    model = InvalidThenValidToolModel()
    tool = StructuredTool.from_function(record)
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool))
    _, state = _plan_and_state()

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(state, 1, "publish")

    execution = asyncio.run(run())

    assert execution.status == "failed"
    assert len(model.calls) == 1


def test_executor_fails_immediately_on_nonzero_atom_exec_result(monkeypatch):
    model = FinalModel(
        json.dumps({"completed": True, "result": "done"}),
        tool_name="exec",
        tool_args={"command": "false"},
    )

    def atom_exec(command: str) -> str:
        """Return a failed Atom command result."""
        return json.dumps({"exit_code": 7, "stdout": "", "stderr": "failed"})

    tool = StructuredTool.from_function(atom_exec, name="exec")
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: ControlledClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", _load_tools(tool))
    _, state = _plan_and_state()

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(state, 1, "publish")

    execution = asyncio.run(run())

    assert execution.status == "failed"
    assert "exit" in execution.error.lower()
    assert len(model.calls) == 1


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
        json.dumps(
            {
                "completed": True,
                "result": "Published the release. Completion criterion met: Release is available",
            }
        ),
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
