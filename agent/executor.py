import json
import os
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, ConfigDict

from memory.state import AgentState, PlanStep, StepExecution


class _StepCompletion(BaseModel):
    """The strictly structured receipt for a completed Plan step."""

    model_config = ConfigDict(extra="forbid")

    completed: Literal[True]
    completion_criterion_met: Literal[True]
    result: str


def _report_step_completion(completed: bool, completion_criterion_met: bool, result: str) -> str:
    """Record the final result for a completed Plan step."""
    return result


class ExecutorGraphState(MessagesState, total=False):
    execution: dict[str, Any]


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
        trace: Any | None = None,
    ) -> None:
        if type(max_rounds) is not int or max_rounds <= 0:
            raise ValueError("max_rounds must be a positive integer")
        self._model = model if model is not None else ChatOpenAI(
            base_url=base_url,
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            model=model_name or os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            max_retries=0,
        )
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
        self._trace = trace
        self._client = None
        self._session = None
        self._tools = None
        self._bound_model = None
        self._completion_model = None
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
            self._active = False

    # one plan step execution
    async def execute(self, state: AgentState, revision: int, step_id: str) -> StepExecution:
        self._require_active()
        step = _resolve_step(state, revision, step_id)
        if step is None:
            return StepExecution(revision, step_id, "failed", error="unknown plan revision or step")

        try:
            self._active_step = step
            self._active_revision = revision
            await self._execute_with_tools()
        except Exception as error:
            return StepExecution(revision, step_id, "failed", error=f"execution failed: {error}")

        return await self._check_step_completion(state, revision, step_id, step)

    async def _check_step_completion(
        self,
        state: AgentState,
        revision: int,
        step_id: str,
        step: PlanStep,
    ) -> StepExecution:
        try:
            messages: list[BaseMessage] = [
                SystemMessage(content=self._INSTRUCTIONS),
                HumanMessage(content=json.dumps(_request_payload(state, revision, step), sort_keys=True)),
            ]
            self._rounds = 0
            config = None
            thread_id = self._thread_id
            if thread_id is None and self._run_id is not None:
                thread_id = f"{self._run_id}:r{revision}:s{step_id}"
            if thread_id is not None:
                config = {"configurable": {"thread_id": thread_id}}
            graph_result = await self._graph.ainvoke({"messages": messages}, config=config)
            tool_messages = [message for message in graph_result["messages"] if isinstance(message, ToolMessage)]
            if tool_messages and not any(_tool_error(message) is None for message in tool_messages):
                return StepExecution(revision, step_id, "failed", error="model did not complete a successful MCP tool call")
            operational_final = graph_result["messages"][-1]
            if isinstance(operational_final, AIMessage) and operational_final.tool_calls:
                return StepExecution(revision, step_id, "failed", error="tool round budget exhausted")
            final = await self._completion_model.ainvoke(
                [
                    *graph_result["messages"],
                    HumanMessage(
                        content=(
                            "Submit the completion receipt now. Explain how the criterion was met "
                            "in result. Set completion_criterion_met to true only if the selected "
                            f"completion criterion was met: {step.completion_criterion}"
                        )
                    ),
                ]
            )
            if self._trace is not None:
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
                return StepExecution(revision, step_id, "failed", error="model did not return a valid completion result")
            return StepExecution(revision, step_id, "completed", result=result)
        except Exception as error:
            if self._checkpointer is not None:
                raise PersistenceError("checkpointed step execution stopped before further tool work") from error
            return StepExecution(revision, step_id, "failed", error=f"execution failed: {error}")

    async def _execute_with_tools(self) -> None:
        if self._tools is not None:
            return
        self._client = MultiServerMCPClient({"atom": self._connection})
        self._session = self._client.session("atom")
        session = await self._session.__aenter__()
        tools = await load_mcp_tools(session)
        if self._tool_allowlist is not None:
            tools = [tool for tool in tools if tool.name in self._tool_allowlist]
        self._tools = tools
        # MCP hosts do not promise OpenAI's strict function-tool schema, so
        # operational tool use remains non-strict. Completion is instead a
        # separate, strictly typed, forced function call after MCP work ends.
        self._bound_model = self._model.bind_tools(tools)
        completion_tool = StructuredTool.from_function(
            _report_step_completion,
            name="report_step_completion",
            args_schema=_StepCompletion,
        )
        self._completion_model = self._model.bind_tools(
            [completion_tool],
            tool_choice="report_step_completion",
            strict=True,
            parallel_tool_calls=False,
        )
        # Tool errors are part of the model/tool conversation: a bad argument or
        # an MCP error must be returned to the model so it can correct its next
        # call, rather than aborting this Step and causing top-level replanning.
        tool_node = ToolNode(tools, handle_tool_errors=True)
        async def call_model(state: MessagesState) -> dict[str, list[BaseMessage]]:
            message = await self._bound_model.ainvoke(state["messages"])
            message = self._with_tool_cwd(message)
            if self._trace is not None and isinstance(message, AIMessage) and message.content:
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
                "execution": {
                    "revision": revision,
                    "step_id": step.id,
                    "status": "running",
                }
            }

        graph.add_node("model", call_model)
        graph.add_node("mark_running", mark_running)
        async def call_tools(state: MessagesState) -> dict[str, list[BaseMessage]]:
            try:
                result = await tool_node.ainvoke(state)
            except Exception as error:
                if self._trace is not None:
                    last = state["messages"][-1]
                    for call in getattr(last, "tool_calls", []):
                        self._trace.tool_result(
                            call.get("name", "unknown"),
                            str(error),
                            error=True,
                        )
                raise
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
            return {**result, "messages": messages}

        graph.add_node("tools", call_tools)
        graph.add_edge(START, "mark_running")
        graph.add_edge("mark_running", "model")
        graph.add_conditional_edges("model", route, {"tools": "tools", END: END})
        graph.add_conditional_edges("tools", route_after_tools, {"model": "model", END: END})
        self._graph = graph.compile(checkpointer=self._checkpointer)

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


def _completion_result(message: AIMessage, completion_criterion: str) -> str | None:
    if not isinstance(message, AIMessage) or len(message.tool_calls) != 1:
        return None
    call = message.tool_calls[0]
    if call.get("name") != "report_step_completion":
        return None
    document = call.get("args")
    if not isinstance(document, dict) or set(document) != {"completed", "completion_criterion_met", "result"}:
        return None
    if (
        document["completed"] is not True
        or document["completion_criterion_met"] is not True
        or not isinstance(document["result"], str)
        or not document["result"].strip()
    ):
        return None
    return document["result"].strip()


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
