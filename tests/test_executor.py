import asyncio
import json
from io import StringIO
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from openai._models import construct_type_unchecked
from openai.types.responses import Response

from agent.executor import Executor, PersistenceError, _executor_model_response, _trace_llm_response
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


@pytest.mark.parametrize("checkpointed", (False, True))
def test_executor_recreates_owned_client_between_plan_steps(monkeypatch, checkpointed):
    clients = []
    requests = []

    def respond(request):
        requests.append(request)
        text = "Published" if len(requests) % 2 else _completed_judgment().content
        return httpx.Response(200, json={
            "id": f"resp_{len(requests)}", "object": "response",
            "created_at": 0, "status": "completed", "model": "model",
            "output": [{"type": "message", "id": "msg_test", "role": "assistant",
                        "status": "completed", "content": [{"type": "output_text",
                        "text": text, "annotations": []}]}],
        })

    def create_client(**kwargs):
        client = AsyncOpenAI(**kwargs, http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(respond), trust_env=False,
        ))
        clients.append(client)
        return client

    async def load_tools(_session):
        return []

    monkeypatch.setattr("llm.llm.AsyncOpenAI", create_client)
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    plan = Plan(1, "Publish twice", (
        PlanStep("first", "Publish first", "Published"),
        PlanStep("second", "Publish second", "Published"),
    ))

    async def run():
        state = AgentState(plan.goal, plan_history=(plan,))
        executor = Executor(
            base_url="https://provider.test/v1", api_key="dummy", model_name="model",
            stream=False, checkpointer=InMemorySaver() if checkpointed else None,
            run_id="client-lifecycle",
        )
        for step in plan.steps:
            async with executor:
                outcome = await executor.execute(state, 1, step.id)
                assert outcome.execution.status == "completed", outcome.execution.error
                state = state.with_step_execution(outcome.execution)
            assert clients[-1].is_closed()
        assert len(clients) == 2
        assert len(requests) == 4

    asyncio.run(run())


def test_executor_leaves_injected_model_open_after_exception_and_reentry():
    class CallerModel(Model):
        async def close(self):
            raise AssertionError("the caller owns this model")

    async def run():
        model = CallerModel()
        executor = Executor(model=model)
        with pytest.raises(RuntimeError, match="step failed"):
            async with executor:
                raise RuntimeError("step failed")
        async with executor:
            assert executor._model is model

    asyncio.run(run())


@pytest.mark.parametrize("checkpointed", (False, True))
@pytest.mark.parametrize("log_level", ("info", "error"))
def test_executor_logs_original_model_failure_before_wrapping(monkeypatch, checkpointed, log_level):
    async def load_tools(_session): return []
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    original = RuntimeError("provider request unavailable")

    class FailingModel(Model):
        async def ainvoke(self, messages):
            raise original

    output = StringIO()

    async def run():
        async with Executor(
            model=FailingModel(),
            checkpointer=InMemorySaver() if checkpointed else None,
            run_id="model-failure",
            trace=RunTrace(output, level=log_level, run_id="model-failure"),
        ) as executor:
            if checkpointed:
                with pytest.raises(PersistenceError) as caught:
                    await executor.execute(_state(), 1, "publish")
                assert caught.value.__cause__ is original
            else:
                outcome = await executor.execute(_state(), 1, "publish")
                assert outcome.execution.error == "execution failed: provider request unavailable"

    asyncio.run(run())
    logged = output.getvalue()
    assert "[model-failure] [execution error] component=executor" in logged
    assert "revision=1 step=publish" in logged
    assert "RuntimeError: provider request unavailable" in logged
    assert "[execution traceback]" in logged
    assert "raise original" in logged


@pytest.mark.parametrize("checkpointed", (False, True))
def test_executor_logs_final_handoff_failure(monkeypatch, checkpointed):
    async def load_tools(_session): return []
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    output = StringIO()
    original = RuntimeError("handoff serialization failed")

    class FailingTrace(RunTrace):
        def llm_complete(self, *args, **kwargs):
            raise original

    async def run():
        async with Executor(
            model=Model(AIMessage(content="Published"), _completed_judgment()),
            checkpointer=InMemorySaver() if checkpointed else None,
            run_id="handoff-failure",
            trace=FailingTrace(output),
        ) as executor:
            if checkpointed:
                with pytest.raises(PersistenceError) as caught:
                    await executor.execute(_state(), 1, "publish")
                assert caught.value.__cause__ is original
            else:
                outcome = await executor.execute(_state(), 1, "publish")
                assert outcome.execution.error == "execution failed: handoff serialization failed"

    asyncio.run(run())
    logged = output.getvalue()
    assert "[execution error] component=executor phase=final handoff" in logged
    assert "revision=1 step=publish" in logged
    assert "RuntimeError: handoff serialization failed" in logged
    assert "[execution traceback]" in logged


def test_executor_returns_freeform_content_without_a_completion_receipt(monkeypatch):
    async def load_tools(session): return []
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    model = Model(
        AIMessage(content="Published successfully.\n\n- Release artifact is available."),
        _completed_judgment("the release artifact is available"),
    )

    async def run():
        async with Executor(model=model) as executor:
            return await executor.execute(_state(), 1, "publish")
    outcome = asyncio.run(run())
    assert outcome.execution.status == "completed"
    assert outcome.execution.result == "Published successfully.\n\n- Release artifact is available."
    assert outcome.context_update is not None
    assert outcome.context_update.files_read == ()
    assert len(model.requests) == 2
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
    model = Model(AIMessage(content=content), _completed_judgment())

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
    )
    output = StringIO()

    async def run():
        async with Executor(model=model, trace=RunTrace(output)) as executor:
            return await executor.execute(_state(), 1, "publish")

    outcome = asyncio.run(run())

    assert outcome.execution.status == "completed"
    assert '"finish_reason": "length"' in output.getvalue()
    assert "[llm complete] revision=1 step=publish" in output.getvalue()


@pytest.mark.parametrize("log_level", ("info", "error"))
def test_executor_trace_prints_usage_and_completed_timing_per_model_request(monkeypatch, log_level):
    async def load_tools(session): return []
    monkeypatch.setattr("agent.executor.MultiServerMCPClient", Client)
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
    usage = {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
    model = Model(
        AIMessage(content="Published", usage_metadata=usage),
        AIMessage(content=_completed_judgment().content, usage_metadata=usage),
    )
    output = StringIO()

    async def run():
        async with Executor(model=model, trace=RunTrace(output, level=log_level)) as executor:
            return await executor.execute(_state(), 1, "publish")

    assert asyncio.run(run()).execution.status == "completed"
    lines = output.getvalue().splitlines()
    usage_lines = [line for line in lines if line.startswith("[llm usage]")]
    timing_lines = [line for line in lines if line.startswith("[llm timing]")]
    assert len(usage_lines) == len(timing_lines) == 2
    assert any("component=executor " in line for line in usage_lines)
    assert any("component=completion_judge " in line for line in usage_lines)
    assert all("request_status=completed" in line for line in timing_lines)


def test_responses_executor_adapter_preserves_provider_usage_for_trace(monkeypatch):
    class ResponsesModel:
        async def complete_response(self, _input, *, tools):
            return SimpleNamespace(
                output_text="Published",
                output=(),
                status="completed",
                usage={"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            )

    monkeypatch.setattr("agent.executor.LLM", ResponsesModel)
    response = asyncio.run(_executor_model_response(ResponsesModel(), [], (), trace=None))

    assert response.response_metadata["usage"] == {
        "input_tokens": 2,
        "output_tokens": 1,
        "total_tokens": 3,
    }
    output = StringIO()
    trace = RunTrace(output, level="error")
    trace.llm_request("executor", {}, static_shape={"request_kind": "tool_round"})
    _trace_llm_response(trace, response)
    assert "[llm usage] component=executor " in output.getvalue()
    assert "[llm timing] component=executor " in output.getvalue()
    assert "request_status=completed" in output.getvalue()


@pytest.mark.parametrize("log_level", ("info", "error"))
def test_responses_executor_tool_message_is_checkpoint_serializable(monkeypatch, log_level):
    # The SDK constructs responses without validation. A validated Pydantic
    # fixture misses the lazy serializer failure of native ResponseUsage.
    native = construct_type_unchecked(type_=Response, value={
        "id": "response-1", "object": "response", "created_at": 1,
        "model": "test", "status": "completed", "parallel_tool_calls": True,
        "tool_choice": "auto", "tools": [],
        "output": [{
            "type": "function_call", "name": "record", "arguments": "{}",
            "call_id": "call-1", "id": "function-1", "status": "completed",
        }],
        "usage": {
            "input_tokens": 2, "output_tokens": 1, "total_tokens": 3,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    })

    class ResponsesModel:
        async def complete_response(self, _input, *, tools):
            return native

    monkeypatch.setattr("agent.executor.LLM", ResponsesModel)
    output = StringIO()
    trace = RunTrace(output, level=log_level)
    executed = []

    def record() -> str:
        """Record the requested action."""
        executed.append(True)
        return "recorded"

    async def run():
        trace.llm_request("executor", {}, static_shape={"request_kind": "tool_round"})
        message = await _executor_model_response(ResponsesModel(), [], (), trace=trace)
        _trace_llm_response(trace, message)
        assert isinstance(message.response_metadata["usage"], dict)
        assert "response" not in message.response_metadata
        graph = StateGraph(MessagesState)
        graph.add_node("tools", ToolNode([StructuredTool.from_function(record)]))
        graph.add_edge(START, "tools")
        graph.add_edge("tools", END)
        graph = graph.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "sdk-serialization"}}
        result = await graph.ainvoke({"messages": [message]}, config=config)
        saved = await graph.aget_state(config)
        assert saved.values["messages"][0].response_metadata == message.response_metadata
        return result

    result = asyncio.run(run())
    assert executed == [True]
    assert result["messages"][-1].content == "recorded"
    logged = output.getvalue()
    assert logged.count("[llm usage]") == logged.count("[llm timing]") == 1
    assert "input_tokens=2 output_tokens=1 total_tokens=3 cached_tokens=0 reasoning_tokens=0" in logged
    assert "request_status=completed" in logged


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
        AIMessage(content="Recorded."),
        _completed_judgment("recorded"),
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

    output = StringIO()

    async def run():
        async with Executor(model=JudgeFailingModel(), checkpointer=InMemorySaver(), run_id="judge-failure", trace=RunTrace(output)) as executor:
            return await executor.execute(_state(), 1, "publish")

    outcome = asyncio.run(run())
    assert outcome.execution.status == "failed"
    assert outcome.execution.error == "completion judge failed: judge unavailable"
    logged = output.getvalue()
    assert "[execution error] component=executor phase=completion judge" in logged
    assert "RuntimeError: judge unavailable" in logged
    assert "CompletionJudgeProviderError: judge unavailable" in logged
    assert "[execution traceback]" in logged
