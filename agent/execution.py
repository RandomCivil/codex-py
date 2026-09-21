import asyncio
import inspect
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI

from agent.configuration import ComponentProviderConfiguration
from agent.model_request import tool_request_shape, trace_llm_context, trace_llm_request
from llm.llm import LLM
from llm.response_format import ResponseFormat, chat_response_format, responses_text_format, require_response_format
from agent.runtime_context import RuntimeContextPolicy
from agent.tool_binding import canonical_mcp_tool_set


ExecutionStatus = Literal["completed", "failed", "blocked"]
ExecutionModeName = Literal["direct", "tool_agent", "react", "plan_execute"]


def conversation_model_input(value: Any) -> Any:
    """Render the public Conversation contract for a model-facing request."""
    if not hasattr(value, "history") or not hasattr(value, "current_input"):
        return value
    return json.dumps(
        {"history": list(value.history), "current_input": value.current_input},
        ensure_ascii=False,
        separators=(",", ":"),
    )


_REACT_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "react_goal_completion",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["answer", "goal_satisfied"],
            "properties": {
                "answer": {"type": "string", "minLength": 1},
                "goal_satisfied": {"const": True},
            },
        },
    },
}

_GENERAL_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["response"],
    "properties": {"response": {"type": "string"}},
}

_DIRECT_JSON_OBJECT_INSTRUCTIONS = "Return exactly one valid JSON object with a response field."


_REACT_TOOL_LOOP_PROMPT = """You are an execution agent. Work iteratively: use an available tool when
evidence or an external action is needed, inspect each result, and correct course when a
tool fails. Do not claim the goal is complete until the available evidence supports it.
When the goal is complete, stop requesting tools so the terminal response can be returned.
"""

_REACT_JSON_OBJECT_PROMPT = _REACT_TOOL_LOOP_PROMPT + """
The terminal response must be exactly one JSON object containing a concise evidence-based
`answer` and `goal_satisfied: true`.

Examples:

User goal: "Find the version in the project configuration."
Assistant: I need project evidence, so I call the available file-reading tool.
Tool result: The configuration reports version 1.4.0.
Assistant: {"answer":"The project version is 1.4.0, as recorded in the configuration.","goal_satisfied":true}

User goal: "Summarize the text supplied in this conversation."
Assistant: {"answer":"Here is the requested summary based on the supplied text.","goal_satisfied":true}

Use the tools and facts for the current goal, never the example values above."""


def _general_chat_response_format(response_format: ResponseFormat) -> dict[str, Any]:
    return chat_response_format(response_format, name="model_response", schema=_GENERAL_RESPONSE_SCHEMA)


def _direct_text_format(response_format: ResponseFormat) -> dict[str, Any]:
    return responses_text_format(
        response_format,
        name="direct_response",
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["response"],
            "properties": {"response": {"type": "string"}},
        },
    )


def _bind_tools_with_response_format(model: Any, tools: Any, response_format: ResponseFormat | None) -> Any:
    """Bind tools and the configured output format at the same model seam."""
    if response_format is None:
        return model.bind_tools(tools)
    try:
        return model.bind_tools(
            tools,
            response_format=_general_chat_response_format(response_format),
        )
    except TypeError as error:
        # Small injected test/double models may expose the older two-argument
        # seam; real ChatOpenAI accepts response_format as a keyword.
        if "response_format" not in str(error) and "unexpected keyword" not in str(error):
            raise
        return model.bind_tools(tools)


def _bind_response_format(model: Any, response_format: dict[str, Any]) -> Any:
    """Apply a terminal response format while retaining lightweight doubles."""
    try:
        return model.bind(response_format=response_format)
    except (AttributeError, TypeError) as error:
        if isinstance(error, TypeError) and "response_format" not in str(error):
            raise
        return model
@dataclass(frozen=True, slots=True)
class ExecutionAnswer:
    """The terminal result shared by whole-task Execution modes."""

    answer: str | None
    status: ExecutionStatus
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"completed", "failed", "blocked"}:
            raise ValueError("execution answer status must be completed, failed, or blocked")
        if self.status == "completed":
            if not isinstance(self.answer, str) or not self.answer.strip():
                raise ValueError("a completed execution answer requires nonempty answer text")
            if self.error is not None:
                raise ValueError("a completed execution answer cannot contain an error")
        elif self.error is None or not self.error.strip():
            raise ValueError("a failed execution answer requires a safe error")


class ExecutionMode(Protocol):
    async def run(self, goal: str) -> ExecutionAnswer: ...


class DirectMode:
    """Obtain one tool-free answer from a language model."""

    def __init__(self, model: Any, *, response_format: ResponseFormat | None = None, trace: Any | None = None) -> None:
        self._model = model
        self._trace = trace
        self._response_format = (
            require_response_format(response_format) if response_format is not None else None
        )

    async def run(self, goal: Any) -> ExecutionAnswer:
        goal = conversation_model_input(goal)
        try:
            request = {
                "input": goal,
                "tools": None,
                **(
                    {"instructions": _DIRECT_JSON_OBJECT_INSTRUCTIONS}
                    if self._response_format == "json_object"
                    else {}
                ),
                **(
                    {"text_format": _direct_text_format(self._response_format)}
                    if self._response_format is not None
                    else {}
                ),
            }
            if not callable(getattr(self._model, "_on_request", None)):
                trace_llm_context(self._trace, request)
            output = "".join(
                [
                    chunk
                    async for chunk in self._model.stream_text(
                        goal,
                        tools=None,
                        **{key: value for key, value in request.items() if key not in {"input", "tools"}},
                    )
                ]
            )
        except Exception:
            return ExecutionAnswer(None, "failed", error="direct model invocation failed")
        if not output.strip():
            return ExecutionAnswer(None, "failed", error="direct model returned an empty response")
        try:
            document = json.loads(output)
        except (TypeError, ValueError, json.JSONDecodeError):
            document = None
        if isinstance(document, dict) and isinstance(document.get("response"), str):
            output = document["response"]
        return ExecutionAnswer(output, "completed")


class ToolRuntime:
    """One invocation-scoped, constrained MCP authority for ephemeral modes."""

    def __init__(
        self,
        *,
        command: str = "poetry",
        args: tuple[str, ...] = ("run", "atom-mcp"),
        cwd: str = "/home/xzp/workspace/atom-mcp",
        tool_cwd: str | None = None,
        env: dict[str, str] | None = None,
        tool_allowlist: tuple[str, ...] | None = None,
        trace: Any | None = None,
    ) -> None:
        self._connection = {
            "transport": "stdio",
            "command": command,
            "args": list(args),
            "cwd": cwd,
            "env": {**os.environ, **(env or {})},
        }
        self._tool_cwd = tool_cwd
        self._allowlist = set(tool_allowlist) if tool_allowlist is not None else None
        self._trace = trace
        self._client = None
        self._session = None
        self._tools: list[Any] | None = None

    async def __aenter__(self) -> "ToolRuntime":
        if self._session is not None:
            raise RuntimeError("Tool runtime is already active")
        self._client = MultiServerMCPClient({"atom": self._connection})
        self._session = self._client.session("atom")
        session = await self._session.__aenter__()
        tools = await load_mcp_tools(session)
        self._tools = [tool for tool in tools if self._allowlist is None or tool.name in self._allowlist]
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if self._session is not None:
                await self._session.__aexit__(exc_type, exc, traceback)
        finally:
            self._client = self._session = None
            self._tools = None

    @property
    def tools(self) -> list[Any]:
        if self._tools is None:
            raise RuntimeError("Tool runtime is not active")
        return self._tools

    async def invoke(self, call: Mapping[str, Any]) -> Any:
        name = call.get("name")
        tool = next((item for item in self.tools if item.name == name), None)
        if tool is None:
            raise ValueError(f"tool {name!r} is not allowed")
        args = dict(call.get("args") or {})
        if self._tool_cwd is not None:
            args["cwd"] = self._tool_cwd
        if self._trace is not None:
            self._trace.tool_call(name, args, call.get("id"))
        result = await tool.ainvoke(args)
        if self._trace is not None:
            self._trace.tool_result(name, result, error=_tool_result_failed(result))
        if _tool_result_failed(result):
            raise _ToolExecutionError("tool returned a non-success result", result)
        return result


class _ToolExecutionError(RuntimeError):
    """Carry a failed tool's original result into the ReAct context policy."""

    def __init__(self, message: str, result: Any) -> None:
        super().__init__(message)
        self.result = result


class ToolAgentMode:
    """Make one model request and, at most, one host tool invocation."""

    def __init__(self, model: Any, tool_runtime: Any, trace: Any | None = None, *, response_format: ResponseFormat | None = None) -> None:
        self._model = model
        self._runtime = tool_runtime
        self._trace = trace
        self._response_format = (
            require_response_format(response_format) if response_format is not None else None
        )

    async def run(self, goal: Any) -> ExecutionAnswer:
        goal = conversation_model_input(goal)
        try:
            async with self._runtime as runtime:
                tools = canonical_mcp_tool_set(runtime.tools)
                bound = _bind_tools_with_response_format(
                    self._model, tools, self._response_format
                )
                trace_llm_request(
                    self._trace,
                    "tool_agent",
                    goal,
                    static_shape={
                        "tools": tool_request_shape(tools),
                        "response_format": self._response_format,
                    },
                )
                response = await bound.ainvoke(goal)
                _trace_llm_response(self._trace, response)
                calls = list(getattr(response, "tool_calls", []) or [])
                if not calls:
                    text = _message_text(response)
                    if not text.strip():
                        return ExecutionAnswer(None, "failed", error="tool-agent model returned an empty response")
                    return ExecutionAnswer(text, "completed")
                for ignored in calls[1:]:
                    if self._trace is not None:
                        _trace_ignored(self._trace, ignored)
                result = await runtime.invoke(calls[0])
                return ExecutionAnswer(render_tool_result(result), "completed")
        except Exception:
            return ExecutionAnswer(None, "failed", error="tool-agent execution failed")


class ReactMode:
    """Iterate model/tool rounds until a structured completion proof is returned."""

    def __init__(
        self,
        model: Any,
        tool_runtime: Any,
        trace: Any | None = None,
        *,
        max_rounds: int = 50,
        response_format: ResponseFormat | None = None,
        context_policy: RuntimeContextPolicy | None = None,
        context_budget: int = 128_000,
        context_model: Any | None = None,
        context_response_format: ResponseFormat | None = None,
    ) -> None:
        if type(max_rounds) is not int or max_rounds <= 0:
            raise ValueError("max_rounds must be a positive integer")
        self._model = model
        self._runtime = tool_runtime
        self._trace = trace
        self._max_rounds = max_rounds
        self._response_format = (
            require_response_format(response_format) if response_format is not None else None
        )
        self._context_policy = context_policy
        self._context_budget = context_budget
        self._context_model = context_model
        self._context_response_format = context_response_format

    def _policy_for_invocation(self) -> RuntimeContextPolicy | Any | None:
        """Return an invocation-local Runtime-context policy."""
        policy = self._context_policy
        if not isinstance(policy, RuntimeContextPolicy):
            return policy
        return policy.fresh_for_invocation()

    async def run(self, goal: Any) -> ExecutionAnswer:
        goal = conversation_model_input(goal)
        try:
            async with self._runtime as runtime:
                # Tool-use rounds must remain unconstrained: some
                # OpenAI-compatible providers reject a response format when
                # it is combined with tool definitions.  The structured
                # completion contract is applied only after tool use stops.
                tools = canonical_mcp_tool_set(runtime.tools)
                model = self._model.bind_tools(tools)
                # Injected doubles can predate the no-tool Observation seam.  The
                # production ChatOpenAI path always enables it; tests and embedders
                # may provide a policy explicitly when their model supports that
                # additional structured request.
                policy = self._policy_for_invocation()
                policy_model = self._context_model or self._model
                if policy is None and (self._context_model is not None or isinstance(self._model, ChatOpenAI)):
                    policy = RuntimeContextPolicy(
                        policy_model, budget=self._context_budget,
                        response_format=self._context_response_format or self._response_format or "json_schema",
                        trace=self._trace,
                    )
                messages: list[Any] = [
                    SystemMessage(
                        content=(
                            _REACT_TOOL_LOOP_PROMPT
                            if self._response_format == "json_schema"
                            else _REACT_JSON_OBJECT_PROMPT
                        )
                    ),
                    HumanMessage(content=goal),
                ]
                for round_number in range(1, self._max_rounds + 1):
                    request_messages = messages
                    if policy is not None:
                        policy.record_model_use()
                        context = await policy.maintain({"goal": goal})
                        # The policy-owned window is the sole historical source
                        # once it is active; old AI/Tool messages must not become
                        # an undocumented second memory channel.
                        request_messages = [messages[0], messages[1], *context.as_messages()]
                    trace_llm_request(
                        self._trace,
                        "react",
                        request_messages,
                        static_shape={
                            "instructions": getattr(request_messages[0], "content", ""),
                            "tools": tool_request_shape(tools),
                            "request_kind": "tool_round",
                        },
                    )
                    response = await model.ainvoke(request_messages)
                    _trace_llm_response(self._trace, response)
                    calls = list(getattr(response, "tool_calls", []) or [])
                    # Once Runtime context is active, its instantaneous
                    # snapshot is the only historical source for subsequent
                    # requests.  Keep the legacy transcript only for the
                    # compatibility path that has no policy.
                    if policy is None:
                        messages.append(response)
                    if not calls:
                        if self._response_format is not None:
                            completion_model = _bind_response_format(
                                self._model,
                                _react_response_format(self._response_format),
                            )
                            completion_messages = [*messages]
                            if policy is not None:
                                policy.record_model_use()
                                context = await policy.maintain({"goal": goal})
                                completion_messages = [messages[0], messages[1], context.as_messages()[0]]
                            final_request = [
                                *completion_messages,
                                HumanMessage(
                                    content=(
                                        "Return the final goal-completion response now."
                                        if self._response_format == "json_schema"
                                        else (
                                            "Return the final goal-completion response now. "
                                            "It must be one JSON object with exactly an evidence-based "
                                            "answer and goal_satisfied: true."
                                        )
                                    )
                                ),
                            ]
                            trace_llm_request(
                                self._trace,
                                "react",
                                final_request,
                                static_shape={
                                    "instructions": getattr(final_request[0], "content", ""),
                                    "terminal_instruction": getattr(final_request[-1], "content", ""),
                                    "request_kind": "goal_completion",
                                    "response_format": _react_response_format(self._response_format),
                                },
                            )
                            response = await completion_model.ainvoke(
                                final_request
                            )
                            _trace_llm_response(self._trace, response)
                            if getattr(response, "tool_calls", []):
                                return ExecutionAnswer(
                                    None,
                                    "failed",
                                    error="react response did not prove goal completion",
                                )
                        return _react_answer(response)
                    if round_number == self._max_rounds:
                        return ExecutionAnswer(None, "failed", error="react round budget exhausted")
                    results = await asyncio.gather(
                        *(self._invoke_for_react(runtime, call) for call in calls)
                    )
                    if policy is not None:
                        await policy.record_tool_round(
                            round_number,
                            calls,
                            [raw_result for _, _, raw_result in results],
                            [content if is_error else None for content, is_error, _ in results],
                        )
                    if policy is None:
                        messages.extend(
                            ToolMessage(
                                content=content,
                                tool_call_id=str(call.get("id") or "unknown"),
                                name=str(call.get("name") or "unknown"),
                                status="error" if is_error else "success",
                            )
                            for call, (content, is_error, _) in zip(calls, results)
                        )
        except Exception:
            return ExecutionAnswer(None, "failed", error="react execution failed")

        return ExecutionAnswer(None, "failed", error="react execution failed")

    async def _invoke_for_react(self, runtime: Any, call: Mapping[str, Any]) -> tuple[str, bool, Any]:
        try:
            result = await runtime.invoke(call)
            return render_tool_result(result), False, result
        except Exception as error:
            if self._trace is not None:
                self._trace.tool_result(
                    call.get("name", "unknown"),
                    str(error),
                    error=True,
                )
            return f"MCP tool failed: {error}", True, getattr(error, "result", {"error": str(error)})


def _react_answer(response: Any) -> ExecutionAnswer:
    try:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            document = json.loads(content)
        elif isinstance(content, Mapping):
            document = content
        else:
            return ExecutionAnswer(None, "failed", error="react response did not prove goal completion")
    except (TypeError, ValueError, json.JSONDecodeError):
        return ExecutionAnswer(None, "failed", error="react response did not prove goal completion")
    if document.get("goal_satisfied") is not True:
        return ExecutionAnswer(None, "failed", error="react response did not prove goal completion")
    answer = document.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return ExecutionAnswer(None, "failed", error="react response did not prove goal completion")
    return ExecutionAnswer(answer, "completed")


def _react_response_format(response_format: ResponseFormat) -> dict[str, Any]:
    if response_format == "json_schema":
        return _REACT_RESPONSE_SCHEMA
    if response_format == "json_object":
        return {"type": "json_object"}
    raise ValueError("response_format must be json_schema or json_object")


def render_tool_result(value: Any) -> str:
    """Render only safely presentable tool values in a stable form."""
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("tool result is empty")
        return value
    if isinstance(value, (Mapping, list, tuple, int, float, bool)) or value is None:
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError) as error:
            raise ValueError("tool result cannot be rendered") from error
    content = getattr(value, "content", None)
    if content is not None:
        return render_tool_result(content)
    raise ValueError("tool result cannot be rendered")


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return render_tool_result(content)


def _tool_result_failed(value: Any) -> bool:
    if isinstance(value, Mapping):
        if value.get("isError") is True or value.get("error"):
            return True
        return isinstance(value.get("returncode"), int) and value["returncode"] != 0
    return getattr(value, "isError", False) is True or (
        isinstance(getattr(value, "returncode", None), int) and value.returncode != 0
    )


def _trace_ignored(trace: Any, call: Mapping[str, Any]) -> None:
    ignored = getattr(trace, "tool_ignored", None)
    if callable(ignored):
        ignored(call.get("name", "unknown"), call.get("id"))
    else:
        trace.tool_call(call.get("name", "unknown"), call.get("args", {}), call.get("id"))


def _trace_llm_response(trace: Any | None, response: Any) -> None:
    """Record a Chat-model response through the same terminal trace contract."""
    if trace is not None:
        trace.llm_response(response)


class PlanExecuteMode:
    """Adapt the existing durable Plan–Execute lifecycle to ExecutionAnswer."""

    def __init__(
        self,
        durable_agent: Any,
        *,
        run_id: str | None = None,
        run_id_factory: Any = uuid.uuid4,
    ) -> None:
        self._durable_agent = durable_agent
        self._run_id = run_id
        self._run_id_factory = run_id_factory

    async def run(self, goal: Any) -> ExecutionAnswer:
        from memory.state import AgentState, deserialize_agent_state

        run_id = self._run_id or str(self._run_id_factory())
        try:
            state = AgentState(goal.current_input) if hasattr(goal, "current_input") else AgentState(goal)
            accepts_conversation_input = (
                "conversation_input" in inspect.signature(self._durable_agent.run).parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in inspect.signature(self._durable_agent.run).parameters.values()
                )
            )
            result = await self._durable_agent.run(
                run_id,
                state,
                **({"conversation_input": goal} if accepts_conversation_input and hasattr(goal, "current_input") else {}),
            )
            status = result.get("status")
            if status == "completed":
                state = result.get("state")
                if isinstance(state, dict):
                    state = deserialize_agent_state(state)
                completed = next(
                    (item for item in reversed(state.step_executions) if item.status == "completed"),
                    None,
                )
                if completed is not None and completed.result:
                    return ExecutionAnswer(completed.result, "completed")
                return ExecutionAnswer(None, "failed", error="plan-execute run completed without a step handoff")
            if status == "blocked":
                return ExecutionAnswer(None, "blocked", error="plan-execute run blocked")
            return ExecutionAnswer(None, "failed", error="plan-execute run did not complete")
        except Exception:
            return ExecutionAnswer(None, "failed", error="plan-execute execution failed")


class _UnavailableMode:
    def __init__(self, mode: ExecutionModeName) -> None:
        self._mode = mode

    async def run(self, goal: str) -> ExecutionAnswer:
        return ExecutionAnswer(
            None,
            "failed",
            error=f"execution mode {self._mode!r} is not implemented",
        )


def create_execution_mode(
    mode: ExecutionModeName,
    *,
    model: Any | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    model_name: str | None = None,
    response_format: ResponseFormat | None = None,
    tool_runtime: Any | None = None,
    trace: Any | None = None,
    durable_agent: Any | None = None,
    run_id: str | None = None,
    max_rounds: int = 50,
    context_budget: int = 128_000,
    context_policy: RuntimeContextPolicy | None = None,
    configuration: ComponentProviderConfiguration | None = None,
) -> ExecutionMode:
    """Select a whole-task Execution mode explicitly.

    Dependencies are injected at this seam so modes remain usable without live
    provider, MCP, or durable-run infrastructure in tests and embeddings.
    """
    if mode not in {"direct", "tool_agent", "react", "plan_execute"}:
        raise ValueError("mode must be direct, tool_agent, react, or plan_execute")
    if mode == "plan_execute":
        if durable_agent is None:
            return _UnavailableMode(mode)
        return PlanExecuteMode(durable_agent, run_id=run_id)
    provider = getattr(configuration, mode, None) if configuration is not None else None
    selected_response_format = (
        (provider.response_format or "json_schema")
        if provider is not None
        else response_format
    )
    if selected_response_format is not None:
        selected_response_format = require_response_format(selected_response_format)
    if provider is not None and model is None:
        if mode == "direct":
            model = LLM(
                provider.base_url,
                provider.api_key,
                provider.model_name,
                response_format=provider.response_format or "json_schema",
                on_event=getattr(trace, "llm_event", None),
                on_request=(
                    (lambda request: trace.llm_request("direct", request))
                    if trace is not None
                    else None
                ),
            )
        else:
            model = ChatOpenAI(
                base_url=provider.base_url,
                api_key=provider.api_key,
                model=provider.model_name,
                max_retries=0,
            )
    if mode == "tool_agent":
        if model is None:
            return _UnavailableMode(mode)
        return ToolAgentMode(
            model,
            tool_runtime or ToolRuntime(trace=trace),
            trace=trace,
            response_format=selected_response_format,
        )
    if mode == "react":
        if model is None:
            return _UnavailableMode(mode)
        context_configuration = configuration.runtime_context if configuration is not None else None
        context_model = None
        if context_configuration is not None:
            context_model = ChatOpenAI(
                base_url=context_configuration.base_url,
                api_key=context_configuration.api_key,
                model=context_configuration.model_name,
                max_retries=0,
            )
        return ReactMode(
            model,
            tool_runtime or ToolRuntime(trace=trace),
            trace=trace,
            max_rounds=max_rounds,
            context_budget=context_budget,
            context_policy=context_policy,
            context_model=context_model,
            context_response_format=(
                context_configuration.response_format if context_configuration is not None else None
            ),
            response_format=selected_response_format,
        )
    if model is None:
        if not all(isinstance(value, str) and value.strip() for value in (base_url, api_key, model_name)):
            raise ValueError("direct mode requires an injected model or explicit provider values")
        model = LLM(
            base_url,
            api_key,
            model_name,
            response_format=selected_response_format,
            on_event=getattr(trace, "llm_event", None),
            on_request=(
                (lambda request: trace.llm_request("direct", request))
                if trace is not None
                else None
            ),
        )
    return DirectMode(model, response_format=selected_response_format, trace=trace)
