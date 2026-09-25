import json
import os
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from llm.line_protocol import LineProtocolError, decode_step_completion_progress

from memory.state import AgentState, ContextUpdate, ExecutionOutcome, PlanStep, StepContext, StepExecution
from agent.runtime_context import (
    RawToolResult,
    RuntimeContext,
    RuntimeContextPolicy,
    observation_decision_context,
)
from agent.model_request import tool_request_shape, trace_llm_request
from agent.tool_binding import canonical_mcp_tool_set
from agent.tool_results import tool_result_failed


class ExecutorGraphState(MessagesState, total=False):
    execution: StepExecution


class PersistenceError(RuntimeError):
    """A checkpointed execution cannot safely continue after an unhandled failure."""


class CompletionJudgeProviderError(RuntimeError):
    """The completion judge provider failed, so this Step must fail closed."""


class Executor:
    """Execute exactly one Plan step through a host-configured MCP tool set."""

    _INSTRUCTIONS = (
        "Use the available MCP tools when they are needed to complete the selected Plan step. "
        "Treat the selected Plan step's completion_criterion as fixed and authoritative. "
        "Before every tool call, check whether the requested step outcome has actually been "
        "produced, the available evidence directly verifies the completion_criterion to a "
        "level proportionate to risk, and no required part of the step remains. "
        "If the criterion is met, stop using tools and return the final handoff; otherwise, "
        "call only a tool that closes a specific remaining gap. Do not repeat an equivalent "
        "inspection or gather extra corroboration after the criterion is met. "
        "If a native tool call is emitted, the host executes it even when the response also has text. "
        "When no native tool call is emitted, put the current Step answer directly in "
        "response content; do not use Line Protocol for that answer."
    )

    def __init__(
        self,
        model: Any | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        stream: bool = False,
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
    ) -> None:
        if type(max_rounds) is not int or max_rounds <= 0:
            raise ValueError("max_rounds must be a positive integer")
        if model is None:
            if not all(isinstance(value, str) and value.strip() for value in (base_url, api_key, model_name)):
                raise ValueError("Executor requires explicit base_url, api_key, and model_name when no model is injected")
            model = ChatOpenAI(
                base_url=base_url,
                api_key=api_key,
                model=model_name,
                streaming=stream,
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
        self._active_context_policy: RuntimeContextPolicy | None = None
        self._active_durable_state: Any = None
        self._client = None
        self._session = None
        self._tools = None
        self._bound_model = None
        self._tool_node = None
        self._tool_graph = None
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
            self._tool_node = None
            self._tool_graph = None
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
            self._active_context_policy = (
                self._context_policy.fresh_for_invocation()
                if self._context_policy is not None
                else None
            )
            policy_model = self._context_model or self._model
            if self._active_context_policy is None and (
                self._context_model is not None or isinstance(self._model, ChatOpenAI)
            ):
                self._active_context_policy = RuntimeContextPolicy(
                    policy_model,
                    budget=self._context_budget,
                    trace=self._trace,
                )
            graph_result = await self._execute_with_tools(
                state, revision, step, step_context, recovery=recovery, attempt=attempt
            )
        except CompletionJudgeProviderError as error:
            return ExecutionOutcome(StepExecution(revision, step_id, "failed", error=f"completion judge failed: {error}"))
        except Exception as error:
            if self._checkpointer is not None:
                raise PersistenceError("checkpointed step execution stopped before further tool work") from error
            return ExecutionOutcome(StepExecution(revision, step_id, "failed", error=f"execution failed: {error}"))

        return self._complete_step(graph_result, revision, step_id)

    def _complete_step(
        self,
        graph_result: dict[str, Any],
        revision: int,
        step_id: str,
    ) -> ExecutionOutcome:
        try:
            if graph_result.get("error"):
                return ExecutionOutcome(StepExecution(revision, step_id, "failed", error=graph_result["error"]))
            operational_final = graph_result["messages"][-1]
            if isinstance(operational_final, AIMessage) and operational_final.tool_calls:
                return ExecutionOutcome(StepExecution(revision, step_id, "failed", error="tool round budget exhausted"))
            if not isinstance(operational_final, AIMessage) or not isinstance(operational_final.content, str):
                return ExecutionOutcome(StepExecution(revision, step_id, "failed", error="model did not return a final text response"))
            handoff = operational_final.content
            if not handoff.strip():
                return ExecutionOutcome(StepExecution(revision, step_id, "failed", error="model did not return a final text response"))
            # Earlier failed ToolMessages are part of the model's correction
            # history. A later no-tool response is the completion proof, even
            # when the failed call was not followed by a successful tool call.
            if self._trace is not None:
                self._trace.llm_complete(
                    revision,
                    step_id,
                    {
                        "content": operational_final.content,
                        "tool_calls": operational_final.tool_calls,
                        "finish_reason": _finish_reason(operational_final),
                    },
                )
            return ExecutionOutcome(
                StepExecution(
                    revision,
                    step_id,
                    "completed",
                    result=handoff,
                    completion_evidence=graph_result.get("completion_evidence"),
                ),
                ContextUpdate(),
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
        """Run operational rounds, judging each settled batch at the host boundary."""
        if self._tools is None:
            self._client = MultiServerMCPClient({"atom": self._connection})
            self._session = self._client.session("atom")
            session = await self._session.__aenter__()
            tools = await load_mcp_tools(session)
            if self._tool_allowlist is not None:
                tools = [tool for tool in tools if tool.name in self._tool_allowlist]
            self._tools = canonical_mcp_tool_set(tools)
            self._bound_model = self._model.bind_tools(self._tools)
            # Preserve the accepted model → ToolNode → model boundary. ToolNode
            # owns MCP invocation and returns recoverable failures as ToolMessages.
            self._tool_node = ToolNode(self._tools, handle_tool_errors=True)
            tool_graph = StateGraph(ExecutorGraphState)
            tool_graph.add_node("tools", self._tool_node)
            tool_graph.add_edge(START, "tools")
            tool_graph.add_edge("tools", END)
            self._tool_graph = tool_graph.compile()
        messages: list[BaseMessage] = [
            SystemMessage(content=self._INSTRUCTIONS),
            HumanMessage(content=json.dumps(_request_payload(state, revision, step), sort_keys=True)),
            HumanMessage(content=_format_step_context(step_context or StepContext())),
        ]
        if recovery:
            messages.insert(
                1,
                HumanMessage(content=(
                    "## Recovery attempt\n\n"
                    "This is a fresh execution attempt after an earlier attempt was interrupted. "
                    "The earlier attempt's messages and tool output are unconfirmed and are not facts. "
                    "Inspect and reconcile the current external state before taking action, then pursue "
                    "the original Plan step completion criterion."
                )),
            )
        if self._checkpointer is not None:
            thread_id = self._thread_id
            if thread_id is None and self._run_id is not None:
                selected_attempt = attempt if attempt is not None else self._attempt
                suffix = f":a{selected_attempt}" if selected_attempt is not None else ""
                thread_id = f"{self._run_id}:r{revision}:s{step.id}{suffix}"
            if thread_id is not None:
                await self._checkpoint_running({"configurable": {"thread_id": thread_id}})
        self._rounds = 0
        criterion_completed = False
        completion_evidence: str | None = None
        failed_tool_results: list[dict[str, Any]] = []
        pending_messages: list[BaseMessage] = []
        while True:
            if criterion_completed:
                handoff_request = [
                    SystemMessage(content=(
                        "The selected Plan step completion criterion is completed by host-validated "
                        "evidence. Return a concise, nonempty final Step handoff. Do not call tools."
                    )),
                    HumanMessage(content=_step_criterion_status(step, True)),
                ]
                handoff_model = self._model.bind_tools(())
                for handoff_attempt in range(2):
                    trace_llm_request(
                        self._trace,
                        "executor",
                        handoff_request,
                        static_shape={
                            "instructions": handoff_request[0].content,
                            "tools": None,
                            "request_kind": "step_handoff" if handoff_attempt == 0 else "step_handoff_repair",
                        },
                    )
                    handoff = await handoff_model.ainvoke(handoff_request)
                    _trace_llm_response(self._trace, handoff)
                    valid_handoff = (
                        not getattr(handoff, "tool_calls", None)
                        and isinstance(getattr(handoff, "content", None), str)
                        and handoff.content.strip()
                    )
                    if valid_handoff:
                        return {"messages": [handoff], "completion_evidence": completion_evidence}
                    if handoff_attempt == 0:
                        handoff_request = [
                            *handoff_request,
                            HumanMessage(content=(
                                "The previous handoff was invalid: return nonempty text only, "
                                "without native tool calls. Re-state the completed Plan-step handoff. "
                                f"Rejected response: {json.dumps(_message_reference(handoff), ensure_ascii=False, default=str)}"
                            )),
                        ]
                return {
                    "messages": [AIMessage(content="")],
                    "error": "model did not return a final text response",
                    "completion_evidence": completion_evidence,
                }
            status = _step_criterion_status(step, criterion_completed)
            request = list(messages)
            if self._active_context_policy is not None:
                self._active_context_policy.record_model_use()
                context = await self._active_context_policy.maintain(self._active_durable_state)
                request = [
                    SystemMessage(content=self._INSTRUCTIONS),
                    *context.as_messages(completion_criteria_status=status),
                    *pending_messages,
                ]
            else:
                request.append(HumanMessage(content="Completion criterion status:\n" + status))
                request.extend(pending_messages)
            pending_messages = []
            trace_llm_request(
                self._trace,
                "executor",
                request,
                static_shape={
                    "instructions": self._INSTRUCTIONS,
                    "tools": tool_request_shape(self._tools),
                    "request_kind": "tool_round",
                },
            )
            response = await self._bound_model.ainvoke(request)
            response = self._with_tool_cwd(response)
            _trace_llm_response(self._trace, response)
            calls = list(getattr(response, "tool_calls", []) or [])
            if calls:
                if criterion_completed:
                    return {"messages": [AIMessage(content="")], "completion_evidence": completion_evidence, "error": "tool call after completion"}
                self._rounds += 1
                if self._trace is not None:
                    for call in calls:
                        self._trace.tool_call(call.get("name", "unknown"), call.get("args", {}), call.get("id"))
                operational_response = response
                graph_response = response.model_copy(update={"content": ""}) if isinstance(response, AIMessage) else response
                results = await self._settle_tool_batch(graph_response)
                for message in results:
                    if isinstance(message, ToolMessage) and self._trace is not None:
                        self._trace.tool_result(getattr(message, "name", "unknown"), message.content, error=_tool_error(message) is not None)
                pending_messages = [
                    message for message in results
                    if isinstance(message, ToolMessage) and _tool_error(message) is not None
                ]
                failed_tool_results.extend(_failed_tool_evidence(calls, results))
                completion_context: list[BaseMessage] = []
                if self._active_context_policy is not None:
                    await self._record_executor_round(operational_response, results)
                    # The settled batch is part of the Runtime context before
                    # its evidence can be trusted by the completion judge.
                    # Refuse to judge when that required context cannot fit;
                    # silently dropping evidence could produce false completion.
                    settled_context = await self._active_context_policy.maintain(self._active_durable_state)
                    completion_context = _completion_context_messages(settled_context)
                successful = [message for message in results if _tool_error(message) is None]
                if successful and not any(_tool_error(message) is not None for message in results):
                    judged = await self._judge_step_completion(
                        step, request, completion_context, operational_response, calls, results,
                        criterion_completed,
                        failed_tool_results,
                    )
                    if judged is not None:
                        criterion_completed, completion_evidence = True, judged
                        continue
                if self._rounds >= self._max_rounds:
                    return {"messages": [AIMessage(content="")], "error": "tool round budget exhausted"}
                if self._active_context_policy is None:
                    messages.extend([graph_response, *results])
                continue

            completion_context = []
            if self._active_context_policy is not None:
                completion_context = _completion_context_messages(
                    await self._active_context_policy.maintain(self._active_durable_state)
                )
            judged = await self._judge_step_completion(
                step, request, completion_context, response, (), (), criterion_completed,
                failed_tool_results,
            )
            if judged is not None:
                criterion_completed, completion_evidence = True, judged
                continue
            self._rounds += 1
            if self._rounds >= self._max_rounds:
                return {"messages": [AIMessage(content="")], "error": "tool round budget exhausted"}
            pending_messages = [HumanMessage(content=(
                "The selected completion criterion is still pending. Continue with the available tools "
                "and address the remaining gap."
            ))]

    async def _judge_step_completion(
        self,
        step: PlanStep,
        request: list[BaseMessage],
        runtime_context: list[BaseMessage],
        response: BaseMessage,
        calls: list[dict[str, Any]],
        results: list[BaseMessage],
        completed: bool,
        prior_failed_tool_results: list[dict[str, Any]],
    ) -> str | None:
        evidence = {
            "selected_step": {
                "id": step.id,
                "intent": step.intent,
                "completion_criterion": step.completion_criterion,
            },
            "criterion_status": "completed" if completed else "pending",
            "operational_request": [_message_reference(item) for item in request],
            "runtime_context_snapshot": [_message_reference(item) for item in runtime_context],
            "operational_response": _message_reference(response),
            "successful_tool_results": [
                {
                    "tool_call_id": str(call.get("id") or "unknown"),
                    "name": str(call.get("name") or "unknown"),
                    "arguments": call.get("args", {}),
                    "raw_result": message.content,
                }
                for call, message in zip(calls, results)
                if _tool_error(message) is None
            ],
            "failed_tool_results": prior_failed_tool_results,
        }
        instructions = (
            "You are a tool-free completion judge for one Plan step. Evaluate only the selected "
            "completion criterion against the supplied execution evidence. Only "
            "runtime_context_snapshot and successful_tool_results are evidentiary; "
            "operational_request and failed_tool_results are provenance or correction context and "
            "must never establish completion. Return exactly one "
            "STEP_COMPLETION_PROGRESS block. Report ALL_COMPLETED=false with no child when the "
            "criterion is not directly proven. Report ALL_COMPLETED=true only with one "
            "COMPLETED_CRITERION numbered 1 and concise directly checkable EVIDENCE. Do not "
            "return ANSWER or prose. Use Line Protocol boundary lines and FIELD=JSON_LITERAL "
            "scalar lines exactly as shown below; do not use XML tags, JSON objects, Markdown "
            "fences, or a colon in place of an equals sign.\n\n"
            "Example when the criterion is not proven:\n"
            "BEGIN STEP_COMPLETION_PROGRESS\n"
            "ALL_COMPLETED=false\n"
            "END STEP_COMPLETION_PROGRESS\n\n"
            "Example when successful evidence directly proves the criterion:\n"
            "BEGIN STEP_COMPLETION_PROGRESS\n"
            "ALL_COMPLETED=true\n"
            "BEGIN COMPLETED_CRITERION\n"
            "NUMBER=1\n"
            'EVIDENCE="The successful tool result confirms the artifact exists."\n'
            "END COMPLETED_CRITERION\n"
            "END STEP_COMPLETION_PROGRESS\n\n"
            "Replace the example evidence with a concise fact from this request. "
            "Return only the applicable block."
        )
        context = [
            SystemMessage(content=instructions),
            HumanMessage(content=json.dumps(evidence, ensure_ascii=False, sort_keys=True, default=str)),
        ]
        judge_model = self._model.bind_tools(())
        trace_llm_request(
            self._trace,
            "completion_judge",
            context,
            static_shape={"instructions": instructions, "tools": None, "request_kind": "completion_judge"},
        )
        try:
            judged = await judge_model.ainvoke(context)
        except Exception as error:
            raise CompletionJudgeProviderError(str(error) or "provider request failed") from error
        _trace_llm_response(self._trace, judged, component="completion_judge")
        try:
            progress, all_completed = decode_step_completion_progress(
                _message_text(judged), completed=completed
            )
            if all_completed:
                return progress[0][1]
            return None
        except LineProtocolError as error:
            if self._trace is not None:
                self._trace.llm_validation_retry("completion_judge", error)
            repair = [
                *context,
                HumanMessage(content=(
                    f"Validation error: {error}\nRejected output (JSON-encoded): "
                    f"{json.dumps(_message_text(judged), ensure_ascii=False)}\n"
                    "Return the corrected STEP_COMPLETION_PROGRESS block only."
                )),
            ]
            trace_llm_request(
                self._trace,
                "completion_judge",
                repair,
                static_shape={"instructions": instructions, "tools": None, "request_kind": "completion_judge_repair"},
            )
            try:
                repaired = await judge_model.ainvoke(repair)
            except Exception as error:
                raise CompletionJudgeProviderError(str(error) or "provider repair request failed") from error
            _trace_llm_response(self._trace, repaired, component="completion_judge")
            try:
                progress, all_completed = decode_step_completion_progress(
                    _message_text(repaired), completed=completed
                )
                return progress[0][1] if all_completed else None
            except LineProtocolError:
                return None
    async def _settle_tool_batch(self, response: BaseMessage) -> list[BaseMessage]:
        """Run one model-selected batch through the Executor's ToolNode boundary."""
        try:
            result = await self._tool_graph.ainvoke({"messages": [response]})
        except Exception as error:
            calls = getattr(response, "tool_calls", []) or []
            return [
                ToolMessage(
                    content=f"MCP tool failed: {error}",
                    tool_call_id=str(call.get("id") or "unknown"),
                    name=str(call.get("name") or "unknown"),
                    status="error",
                )
                for call in calls
            ]
        return [
            _as_error_tool_message(message)
            for message in result.get("messages", [])
            if isinstance(message, ToolMessage)
        ]

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
        step = self._active_step
        await self._active_context_policy.record_tool_round(
            self._rounds,
            response.tool_calls,
            [message.content for message in results],
            [_tool_error(message) for message in results],
            decision_context=observation_decision_context(
                goal=self._active_durable_state["agent_state"]["goal"],
                execution_mode="plan_execute",
                plan_step={
                    "revision": self._active_revision,
                    "id": step.id,
                    "intent": step.intent,
                    "completion_criterion": step.completion_criterion,
                },
                completion_criterion=step.completion_criterion,
            ),
            suppress_observations=any(_tool_error(message) is not None for message in results),
        )

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
                "completion_evidence": execution.completion_evidence,
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


def _finish_reason(message: AIMessage) -> Any:
    metadata = getattr(message, "response_metadata", None)
    if not isinstance(metadata, dict):
        return None
    return metadata.get("finish_reason") or metadata.get("stop_reason")


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


def _step_criterion_status(step: PlanStep, completed: bool) -> str:
    return f"1. [{'completed' if completed else 'pending'}] {step.completion_criterion}"


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)


def _message_reference(message: Any) -> dict[str, Any]:
    reference = {
        "message_type": getattr(message, "type", type(message).__name__),
        "content": _message_text(message),
    }
    for attribute in ("tool_calls", "tool_call_id", "name", "status"):
        value = getattr(message, attribute, None)
        if value is not None:
            reference[attribute] = value
    return reference


def _completion_context_messages(context: RuntimeContext) -> list[BaseMessage]:
    """Render a settled Runtime snapshot without failed calls as evidence."""
    evidence_only = RuntimeContext(
        context.durable_state,
        context.observations,
        tuple(
            RawToolResult(item.round, tuple(call for call in item.calls if call.error is None))
            for item in context.raw_tool_results
            if any(call.error is None for call in item.calls)
        ),
    )
    return evidence_only.as_messages()


def _failed_tool_evidence(calls: list[dict[str, Any]], results: list[BaseMessage]) -> list[dict[str, Any]]:
    return [
        {
            "tool_call_id": str(call.get("id") or "unknown"),
            "name": str(call.get("name") or "unknown"),
            "error": _tool_error(message),
        }
        for call, message in zip(calls, results)
        if _tool_error(message) is not None
    ]


def _trace_llm_response(trace: Any | None, response: Any, *, component: str | None = None) -> None:
    if trace is None:
        return
    callback = getattr(trace, "llm_response", None)
    if callback is None:
        return
    if component is None:
        callback(response)
    else:
        try:
            callback(response, component=component)
        except TypeError:
            callback(response)


def _tool_error(message: BaseMessage) -> str | None:
    if not isinstance(message, ToolMessage):
        return None
    if getattr(message, "status", None) == "error":
        return f"MCP tool failed: {message.content}"
    if tool_result_failed(message.content):
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
