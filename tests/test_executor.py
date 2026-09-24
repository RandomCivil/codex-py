import asyncio
import json
from io import StringIO

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from agent.executor import Executor
from agent.trace import RunTrace
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


def _completed_judgment(evidence="the criterion is satisfied"):
    return AIMessage(content=(
        "BEGIN STEP_COMPLETION_PROGRESS\nALL_COMPLETED=true\n"
        "BEGIN COMPLETED_CRITERION\nNUMBER=1\n"
        f"EVIDENCE=\"{evidence}\"\nEND COMPLETED_CRITERION\n"
        "END STEP_COMPLETION_PROGRESS"
    ))


def _state():
    plan = Plan(1, "Publish", (PlanStep("publish", "Publish", "Published"),))
    return AgentState("Publish", plan_history=(plan,))


def test_executor_returns_freeform_content_without_a_completion_receipt(monkeypatch):
    async def load_tools(session): return []
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = Model(
        AIMessage(content="Published successfully.\n\n- Release artifact is available."),
        _completed_judgment("the release artifact is available"),
        AIMessage(content="Published successfully.\n\n- Release artifact is available."),
    )

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(_state(), 1, "publish")
    outcome = asyncio.run(run())
    assert outcome.execution.status == "completed"
    assert outcome.execution.result == "Published successfully.\n\n- Release artifact is available."
    assert outcome.context_update is not None
    assert outcome.context_update.files_read == ()
    assert len(model.requests) == 3
    assert "NO_TOOL" not in model.requests[0][0].content
    assert "STEP_COMPLETION" not in model.requests[0][0].content
    assert "completion_criterion as fixed and authoritative" in model.requests[0][0].content
    assert "Before every tool call" in model.requests[0][0].content
    assert "Do not repeat an equivalent inspection" in model.requests[0][0].content


def test_executor_returns_content_verbatim_without_line_protocol_parsing(monkeypatch):
    async def load_tools(session): return []
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    content = "BEGIN STEP_COMPLETION\nthis is ordinary text\nEND STEP_COMPLETION"
    model = Model(AIMessage(content=content), _completed_judgment(), AIMessage(content=content))

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(_state(), 1, "publish")
    outcome = asyncio.run(run())
    assert outcome.execution.result == content


def test_executor_traces_finish_reason_for_the_terminal_response(monkeypatch):
    async def load_tools(session): return []
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = Model(
        AIMessage(content="Published", response_metadata={"finish_reason": "length"}),
        _completed_judgment(),
        AIMessage(content="Published"),
    )
    output = StringIO()

    async def run():
        async with Executor(model=model, trace=RunTrace(output)) as executor:
            return await executor.execute(_state(), 1, "publish")

    outcome = asyncio.run(run())

    assert outcome.execution.status == "completed"
    assert '"finish_reason": "length"' in output.getvalue()
    assert "[llm complete] revision=1 step=publish" in output.getvalue()


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
        _completed_judgment("recorded"),
        AIMessage(content="Recorded."),
    )

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(_state(), 1, "publish")
    outcome = asyncio.run(run())
    assert outcome.execution.status == "completed", outcome.execution.error
    assert outcome.execution.result == "Recorded."
    assert len(model.requests) == 3
    assert "completion_criterion as fixed and authoritative" in model.requests[0][0].content


def test_judge_excludes_failed_tool_output_from_its_runtime_evidence(monkeypatch):
    def reject() -> str:
        """Fail so the model can correct its next action."""
        raise RuntimeError("patch rejected")

    async def load_tools(_session): return [StructuredTool.from_function(reject, name="apply_patch")]
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = Model(
        AIMessage(content="", tool_calls=[{"name": "apply_patch", "args": {}, "id": "reject-1"}]),
        AIMessage(content="Recovered after correction."),
        _completed_judgment("the correction is complete"),
        AIMessage(content="Recovered after correction."),
    )

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(_state(), 1, "publish")

    assert asyncio.run(run()).execution.status == "completed"
    judge_evidence = json.loads(model.requests[2][-1].content)
    snapshot = judge_evidence["runtime_context_snapshot"]
    assert "patch rejected" not in json.dumps(snapshot)
    failed = judge_evidence["failed_tool_results"]
    assert failed[0]["tool_call_id"] == "reject-1"
    assert failed[0]["name"] == "apply_patch"
    assert "patch rejected" in failed[0]["error"]


def test_judge_provider_failure_returns_a_failed_step_with_a_checkpointer(monkeypatch):
    async def load_tools(_session): return []
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)

    class JudgeFailingModel(Model):
        async def ainvoke(self, messages):
            self.requests.append(messages)
            if "tool-free completion judge" in messages[0].content:
                raise RuntimeError("judge unavailable")
            return AIMessage(content="Published successfully.")

    async def run():
        async with Executor(model=JudgeFailingModel(), checkpointer=InMemorySaver(), run_id="judge-failure") as executor:
            return await executor.execute(_state(), 1, "publish")

    outcome = asyncio.run(run())
    assert outcome.execution.status == "failed"
    assert outcome.execution.error == "completion judge failed: judge unavailable"
