import json
import os
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, ConfigDict, Field

from llm.response_format import ResponseFormat, require_response_format
from memory.state import AgentState, ContextUpdate, ExecutionOutcome, PlanStep, StepContext, StepExecution
from agent.runtime_context import RuntimeContextPolicy


class _StepCompletion(BaseModel):
    """The JSON Schema handoff receipt for a completed Plan step."""

    model_config = ConfigDict(extra="forbid")

    completed: Literal[True]
    completion_criterion_met: Literal[True]
    result: str = Field(
        min_length=1,
        description=(
            "A concise handoff and evidence summary for the next Plan step. "
            "State the material outcome and the supporting tool output, artifact, "
            "identifier, or state fact; do not reproduce the message transcript."
        )
    )
    files_read: list[str]
    files_modified: list[str]
    observations: list[str]


_COMPLETION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "step_completion",
        "strict": True,
        "schema": _StepCompletion.model_json_schema(),
    },
}


def _completion_response_format(response_format: ResponseFormat) -> dict[str, Any]:
    if response_format == "json_schema":
        return _COMPLETION_RESPONSE_FORMAT
    if response_format == "json_object":
        return {"type": "json_object"}
    raise ValueError("response_format must be json_schema or json_object")


class ExecutorGraphState(MessagesState, total=False):
    execution: StepExecution


class PersistenceError(RuntimeError):
    """A checkpointed execution cannot safely continue after an unhandled failure."""


class Executor:
    """Execute exactly one Plan step through a host-configured MCP tool set."""

    _INSTRUCTIONS = (
        "Use the available MCP tools when they are needed to complete the selected Plan step. "
        "If no tool use is needed, stop requesting MCP tools. The Executor "
        "will collect the required structured completion receipt separately."
    )

    def __init__(
        self,
        model: Any | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        response_format: ResponseFormat = "json_schema",
        command: str = "poetry",
        args: tuple[str, ...] = ("run", "atom-mcp"),
        cwd: str = "/home/xzp/workspace/atom-mcp",
        tool_cwd: str | None = None,
        env: dict[str, str] | None = None,
        tool_allowlist: tuple[str, ...] | None = None,
        max_rounds: int = 50,
        checkpointer: Any | None = None,
        thread_id: str | None = None,
        run_id: str | None = None,
        attempt: int | None = None,
        trace: Any | None = None,
        context_policy: RuntimeContextPolicy | None = None,
        context_budget: int = 128_000,
        context_model: Any | None = None,
        context_response_format: ResponseFormat | None = None,
    ) -> None:
        if type(max_rounds) is not int or max_rounds <= 0:
            raise ValueError("max_rounds must be a positive integer")
        self._response_format = require_response_format(response_format)
        if model is None:
            if not all(isinstance(value, str) and value.strip() for value in (base_url, api_key, model_name)):
                raise ValueError("Executor requires explicit base_url, api_key, and model_name when no model is injected")
            model = ChatOpenAI(
                base_url=base_url,
                api_key=api_key,
                model=model_name,
                max_retries=0,
            )
        self._model = model
        self._connection = {
            "transport": "stdio",
            "command": command,
            "args": list(args),
            "cwd": cwd,
            "env": {**os.environ, **(env or {})},
        }
        self._tool_cwd = tool_cwd
        self._tool_allowlist = set(tool_allowlist) if tool_allowlist is not None else None
        self._max_rounds = max_rounds
        if checkpointer is not None and thread_id is None and run_id is None:
            raise ValueError("a checkpointer requires thread_id or run_id")
        if thread_id is not None and run_id is not None:
            raise ValueError("thread_id and run_id are mutually exclusive")
        self._checkpointer = checkpointer
        self._thread_id = thread_id
        self._run_id = run_id
        self._attempt = attempt
        self._trace = trace
        self._context_policy = context_policy
        self._context_budget = context_budget
        self._context_model = context_model
        self._context_response_format = context_response_format
        self._active_context_policy: RuntimeContextPolicy | None = None
        self._active_durable_state: Any = None
        self._client = None
        self._session = None
        self._tools = None
        self._bound_model = None
        self._completion_model = None
        self._graph = None
        self._active = False

    async def __aenter__(self) -> "Executor":
        if self._active:
            raise RuntimeError("Executor is already active")
        self._active = True
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        try:
            if self._session is not None:
                await self._session.__aexit__(exc_type, exc, traceback)
        finally:
            self._client = None
            self._session = None
            self._tools = None
            self._bound_model = None
            self._completion_model = None
            self._graph = None
            self._active = False

    # one plan step execution
    async def execute(
        self,
        state: AgentState,
        revision: int,
        step_id: str,
        step_context: StepContext | None = None,
        *,
        recovery: bool = False,
        attempt: int | None = None,
    ) -> ExecutionOutcome:
        self._require_active()
        step = _resolve_step(state, revision, step_id)
        if step is None:
            return ExecutionOutcome(StepExecution(revision, step_id, "failed", error="unknown plan revision or step"))

        try:
            self._active_step = step
            self._active_revision = revision
            self._active_durable_state = _durable_context_payload(state, revision, step, step_context)
            self._active_context_policy = self._context_policy
            policy_model = self._context_model or self._model
            if self._active_context_policy is None and (
                self._context_model is not None or isinstance(self._model, ChatOpenAI)
            ):
                self._active_context_policy = RuntimeContextPolicy(
                    policy_model,
                    budget=self._context_budget,
                    response_format=self._context_response_format or self._response_format,
                    trace=self._trace,
                )
            graph_result = await self._execute_with_tools(
                state, revision, step, step_context, recovery=recovery, attempt=attempt
            )
        except Exception as error:
            if self._checkpointer is not None:
                raise PersistenceError("checkpointed step execution stopped before further tool work") from error
            return ExecutionOutcome(StepExecution(revision, step_id, "failed", error=f"execution failed: {error}"))

        return await self._check_step_completion(graph_result, revision, step_id, step)

    async def _check_step_completion(
        self,
        graph_result: dict[str, Any],
        revision: int,
        step_id: str,
        step: PlanStep,
    ) -> ExecutionOutcome:
        try:
            tool_messages = [message for message in graph_result["messages"] if isinstance(message, ToolMessage)]
            if tool_messages and not any(_tool_error(message) is None for message in tool_messages):
                return ExecutionOutcome(StepExecution(revision, step_id, "failed", error="model did not complete a successful MCP tool call"))
            operational_final = graph_result["messages"][-1]
            if isinstance(operational_final, AIMessage) and operational_final.tool_calls:
                return ExecutionOutcome(StepExecution(revision, step_id, "failed", error="tool round budget exhausted"))
            completion_prompt = (
                "Submit the JSON completion receipt now. `result` is the handoff and "
                "evidence summary that subsequent Plan steps will receive. Derive it "
                "only from the preceding messages: state the material outcome and the "
                "supporting tool output, artifact path, resource identifier, or state "
                "fact. Be concise; do not merely say it completed or reproduce the "
                "message transcript. Return one top-level JSON object directly with "
                "exactly these fields: completed, completion_criterion_met, result, "
                "files_read, files_modified, and observations. Do not wrap the receipt "
                'in `{"type":"json_object","result":...}`; `json_object` is only the '
                "response format, not a field in your response. Do not include any other "
                "fields, including step_id, revision, status, intent, artifact, outcome, "
                "or type. Set completed and completion_criterion_met to true only if the "
                "selected completion criterion was met: "
                f"{step.completion_criterion}"
            )
            if self._response_format == "json_object":
                completion_prompt += (
                    " Follow this example exactly in shape (use facts from the preceding "
                    "messages, not these example values): "
                    '{"completed":true,"completion_criterion_met":true,"result":"The '
                    'selected outcome is complete; supporting evidence is recorded.",'
                    '"files_read":["docs/example.md"],"files_modified":["src/example.py"],'
                    '"observations":["The completion criterion is satisfied."]}'
                )
            final = await self._completion_model.ainvoke(
                await self._completion_messages(graph_result["messages"], completion_prompt)
            )
            if self._trace is not None:
                self._trace.llm_response(final)
                self._trace.llm_complete(
                    revision,
                    step_id,
                    {
                        "content": getattr(final, "content", None),
                        "tool_calls": getattr(final, "tool_calls", []),
                    },
                )
            result = _completion_result(final, step.completion_criterion)
            if result is None:
                return ExecutionOutcome(StepExecution(revision, step_id, "failed", error="model did not return a valid completion result"))
            handoff, context_update = result
            return ExecutionOutcome(
                StepExecution(revision, step_id, "completed", result=handoff),
                context_update,
            )
        except Exception as error:
            if self._checkpointer is not None:
                raise PersistenceError("checkpointed step execution stopped before further tool work") from error
            return ExecutionOutcome(StepExecution(revision, step_id, "failed", error=f"execution failed: {error}"))

    async def _execute_with_tools(
        self,
        state: AgentState,
        revision: int,
        step: PlanStep,
        step_context: StepContext | None = None,
        *,
        recovery: bool = False,
        attempt: int | None = None,
    ) -> dict[str, Any]:
        if self._tools is None:
            self._client = MultiServerMCPClient({"atom": self._connection})
            self._session = self._client.session("atom")
            session = await self._session.__aenter__()
            tools = await load_mcp_tools(session)
            if self._tool_allowlist is not None:
                tools = [tool for tool in tools if tool.name in self._tool_allowlist]
            self._tools = tools
            # MCP hosts do not promise OpenAI's strict function-tool schema, so
            # operational tool use remains non-strict. Completion is instead a
            # separate structured response after MCP work ends; local parsing
            # and invariant validation apply in both provider modes.
            self._bound_model = self._model.bind_tools(tools)
            self._completion_model = self._model.bind(response_format=_completion_response_format(self._response_format))
            # Tool errors are part of the model/tool conversation: a bad argument or
            # an MCP error must be returned to the model so it can correct its next
            # call, rather than aborting this Step and causing top-level replanning.
            tool_node = ToolNode(tools, handle_tool_errors=True)
        else:
            tool_node = ToolNode(self._tools, handle_tool_errors=True)

        async def call_model(state: MessagesState) -> dict[str, list[BaseMessage]]:
            request = state["messages"]
            if self._active_context_policy is not None:
                self._active_context_policy.record_model_use()
                context = await self._active_context_policy.maintain(self._active_durable_state)
                request = [SystemMessage(content=self._INSTRUCTIONS), *context.as_messages()]
            _trace_llm_context(self._trace, request)
            message = await self._bound_model.ainvoke(request)
            message = self._with_tool_cwd(message)
            if self._trace is not None and isinstance(message, AIMessage):
                self._trace.llm_response(message)
                if message.content:
                    self._trace.llm_text("output", message.content)
            if isinstance(message, AIMessage) and message.tool_calls:
                self._rounds += 1
                if self._trace is not None:
                    for call in message.tool_calls:
                        self._trace.tool_call(call.get("name", "unknown"), call.get("args", {}), call.get("id"))
            return {"messages": [message]}

        def route(state: MessagesState) -> str:
            last = state["messages"][-1]
            if not isinstance(last, AIMessage) or not last.tool_calls:
                return END
            return "tools" if self._rounds <= self._max_rounds else END

        def route_after_tools(state: MessagesState) -> str:
            return "model"

        graph = StateGraph(ExecutorGraphState)
        async def mark_running(state: ExecutorGraphState) -> ExecutorGraphState:
            step = self._active_step
            revision = self._active_revision
            return {
                "execution": StepExecution(revision, step.id, "running")
            }

        graph.add_node("model", call_model)
        graph.add_node("mark_running", mark_running)
        async def call_tools(state: MessagesState) -> dict[str, list[BaseMessage]]:
            try:
                result = await tool_node.ainvoke(state)
            except Exception as error:
                last = state["messages"][-1]
                messages = [
                    ToolMessage(
                        content=f"MCP tool failed: {error}",
                        tool_call_id=str(call.get("id") or "unknown"),
                        name=str(call.get("name") or "unknown"),
                        status="error",
                    )
                    for call in getattr(last, "tool_calls", [])
                ]
                if self._trace is not None:
                    for call in getattr(last, "tool_calls", []):
                        self._trace.tool_result(
                            call.get("name", "unknown"),
                            str(error),
                            error=True,
                        )
                # A tool failure is actionable model context, not a reason for
                # the Executor to end this step and make the Agent replan.
                if self._active_context_policy is not None:
                    await self._record_executor_round(state["messages"][-1], messages)
                return {"messages": messages}
            messages = [
                _as_error_tool_message(message)
                for message in result.get("messages", [])
            ]
            for message in messages:
                if isinstance(message, ToolMessage) and self._trace is not None:
                    self._trace.tool_result(
                        getattr(message, "name", "unknown"),
                        message.content,
                        error=_tool_error(message) is not None,
                    )
            if self._active_context_policy is not None:
                await self._record_executor_round(state["messages"][-1], messages)
            return {**result, "messages": messages}

        graph.add_node("tools", call_tools)
        graph.add_edge(START, "mark_running")
        graph.add_edge("mark_running", "model")
        graph.add_conditional_edges("model", route, {"tools": "tools", END: END})
        graph.add_conditional_edges("tools", route_after_tools, {"model": "model", END: END})
        if self._graph is None:
            # Tool messages are transient Runtime context.  The checkpointed
            # graph therefore deliberately has no saver; only the tiny
            # pre-work marker below is persisted for recovery diagnostics.
            self._graph = graph.compile(checkpointer=None)

        messages: list[BaseMessage] = [
            SystemMessage(content=self._INSTRUCTIONS),
            HumanMessage(content=json.dumps(_request_payload(state, revision, step), sort_keys=True)),
            HumanMessage(content=_format_step_context(step_context or StepContext())),
        ]
        if recovery:
            messages.insert(
                1,
                HumanMessage(
                    content=(
                        "## Recovery attempt\n\n"
                        "This is a fresh execution attempt after an earlier attempt was interrupted. "
                        "The earlier attempt's messages and tool output are unconfirmed and are not facts. "
                        "Inspect and reconcile the current external state before taking action, then pursue "
                        "the original Plan step completion criterion."
                    )
                ),
            )
        self._rounds = 0
        config = None
        thread_id = self._thread_id
        if thread_id is None and self._run_id is not None:
            selected_attempt = attempt if attempt is not None else self._attempt
            suffix = f":a{selected_attempt}" if selected_attempt is not None else ""
            thread_id = f"{self._run_id}:r{revision}:s{step.id}{suffix}"
        if thread_id is not None:
            config = {"configurable": {"thread_id": thread_id}}
        if self._checkpointer is not None and config is not None:
            await self._checkpoint_running(config)
        return await self._graph.ainvoke({"messages": messages}, config=config)

    async def _checkpoint_running(self, config: dict[str, Any]) -> None:
        graph = StateGraph(ExecutorGraphState)

        async def mark_running(_: ExecutorGraphState) -> ExecutorGraphState:
            return {
                "execution": StepExecution(
                    self._active_revision,
                    self._active_step.id,
                    "running",
                )
            }

        graph.add_node("running", mark_running)
        graph.add_edge(START, "running")
        graph.add_edge("running", END)
        await graph.compile(checkpointer=self._checkpointer).ainvoke({}, config=config)

    async def _record_executor_round(self, response: BaseMessage, results: list[BaseMessage]) -> None:
        if not isinstance(response, AIMessage):
            return
        await self._active_context_policy.record_tool_round(
            self._rounds,
            response.tool_calls,
            [message.content for message in results],
            [_tool_error(message) for message in results],
        )

    async def _completion_messages(self, messages: list[BaseMessage], prompt: str) -> list[BaseMessage]:
        result = [*messages]
        if self._active_context_policy is not None:
            self._active_context_policy.record_model_use()
            # The final receipt sees the same Durable State and layered evidence
            # as the last operational request; it never receives trace data.
            context = await self._active_context_policy.maintain(self._active_durable_state)
            result = [SystemMessage(content=self._INSTRUCTIONS), context.as_messages()[0]]
        result.append(HumanMessage(content=prompt))
        _trace_llm_context(self._trace, result)
        return result

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("Executor must be used as an asynchronous context manager")

    def _with_tool_cwd(self, message: BaseMessage) -> BaseMessage:
        """Apply the host-selected working directory to every Atom tool call."""
        if self._tool_cwd is None or not isinstance(message, AIMessage) or not message.tool_calls:
            return message
        tool_calls = [
            {
                **call,
                "args": {
                    **(call["args"] if isinstance(call.get("args"), dict) else {}),
                    "cwd": self._tool_cwd,
                },
            }
            for call in message.tool_calls
        ]
        return message.model_copy(update={"tool_calls": tool_calls})


def _resolve_step(state: AgentState, revision: int, step_id: str) -> PlanStep | None:
    for plan in state.plan_history:
        if plan.revision == revision:
            return next((step for step in plan.steps if step.id == step_id), None)
    return None


def _request_payload(state: AgentState, revision: int, step: PlanStep) -> dict[str, Any]:
    return {
        "goal": state.goal,
        "selected_plan_step": {
            "revision": revision,
            "id": step.id,
            "intent": step.intent,
            "completion_criterion": step.completion_criterion,
        },
        "memory_summary": state.memory_summary,
        "plan_history": [
            {
                "revision": plan.revision,
                "goal": plan.goal,
                "steps": [
                    {
                        "id": item.id,
                        "intent": item.intent,
                        "completion_criterion": item.completion_criterion,
                    }
                    for item in plan.steps
                ],
            }
            for plan in state.plan_history
        ],
        "step_executions": [
            {
                "revision": execution.revision,
                "step_id": execution.step_id,
                "status": execution.status,
                "result": execution.result,
                "error": execution.error,
            }
            for execution in state.step_executions
        ],
    }


def _durable_context_payload(
    state: AgentState,
    revision: int,
    step: PlanStep,
    step_context: StepContext | None,
) -> dict[str, Any]:
    """Serialize the durable execution inputs without leaking transient graph state."""
    context = step_context or StepContext()
    return {
        "agent_state": _request_payload(state, revision, step),
        "step_context": {
            "files_read": list(context.files_read),
            "files_modified": list(context.files_modified),
            "observations": list(context.observations),
        },
    }


def _completion_result(
    message: AIMessage, completion_criterion: str
) -> tuple[str, ContextUpdate] | None:
    if not isinstance(message, AIMessage) or not isinstance(message.content, str):
        return None
    try:
        document = json.loads(message.content)
    except json.JSONDecodeError:
        return None
    required = {
        "completed",
        "completion_criterion_met",
        "result",
        "files_read",
        "files_modified",
        "observations",
    }
    if not isinstance(document, dict) or set(document) != required:
        return None
    if (
        document["completed"] is not True
        or document["completion_criterion_met"] is not True
        or not isinstance(document["result"], str)
        or not document["result"].strip()
        or not isinstance(document["files_read"], list)
        or not isinstance(document["files_modified"], list)
        or not isinstance(document["observations"], list)
    ):
        return None
    try:
        context_update = ContextUpdate(
            files_read=document["files_read"],
            files_modified=document["files_modified"],
            observations=document["observations"],
        )
    except (TypeError, ValueError):
        return None
    return document["result"].strip(), context_update


def _format_step_context(context: StepContext) -> str:
    def section(title: str, values: tuple[str, ...]) -> str:
        items = "\n".join(f"- {value}" for value in values) or "- (none)"
        return f"### {title}\n{items}"

    return "\n\n".join(
        (
            "## Step context",
            section("Files read", context.files_read),
            section("Files modified", context.files_modified),
            section("Observations", context.observations),
        )
    )


def _tool_error(message: BaseMessage) -> str | None:
    if not isinstance(message, ToolMessage):
        return None
    if getattr(message, "status", None) == "error":
        return f"MCP tool failed: {message.content}"
    if getattr(message, "name", None) in {"exec", "atom.exec"}:
        exit_code = _exit_code(message.content)
        if exit_code not in (None, 0):
            return f"Atom exec failed with exit code {exit_code}"
    return None


def _as_error_tool_message(message: BaseMessage) -> BaseMessage:
    """Mark nonzero Atom exec results as errors before returning them to the model."""
    if isinstance(message, ToolMessage) and _tool_error(message) is not None:
        return message.model_copy(update={"status": "error"})
    return message


def _exit_code(content: Any) -> int | None:
    value: Any = content
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, dict):
        for key in ("exit_code", "exitCode", "returncode"):
            if key in value:
                try:
                    return int(value[key])
                except (TypeError, ValueError):
                    return None
    return None


def _trace_llm_context(trace: Any | None, context: Any) -> None:
    if trace is not None:
        callback = getattr(trace, "llm_context", None)
        if callback is not None:
            callback(context)
