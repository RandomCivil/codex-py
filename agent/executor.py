import json
import os
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from memory.state import AgentState, PlanStep, StepExecution


class ExecutorGraphState(MessagesState, total=False):
    execution: dict[str, Any]


class PersistenceError(RuntimeError):
    """A checkpointed execution cannot safely continue after an unhandled failure."""


class Executor:
    """Execute exactly one Plan step through a host-configured MCP tool set."""

    _INSTRUCTIONS = (
        "Use the available MCP tools to complete the selected Plan step. "
        "After successful tool use, return only strict JSON with exactly these "
        "fields: completed and result. completed must be true and result must "
        "briefly explain how the completion criterion was met. End result with "
        "'Completion criterion met: <the exact selected completion criterion>'; "
        "include a non-empty explanation before that statement."
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
        env: dict[str, str] | None = None,
        tool_allowlist: tuple[str, ...] | None = None,
        max_rounds: int = 10,
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
            self._active = False

    async def execute(self, state: AgentState, revision: int, step_id: str) -> StepExecution:
        self._require_active()
        step = _resolve_step(state, revision, step_id)
        if step is None:
            return StepExecution(revision, step_id, "failed", error="unknown plan revision or step")

        try:
            self._active_step = step
            self._active_revision = revision
            await self._ensure_tools()
        except Exception as error:
            return StepExecution(revision, step_id, "failed", error=f"execution failed: {error}")

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
            tool_error = next((error for message in tool_messages if (error := _tool_error(message))), None)
            if tool_error is not None:
                return StepExecution(revision, step_id, "failed", error=tool_error)
            if not tool_messages:
                return StepExecution(revision, step_id, "failed", error="model did not use an MCP tool")
            final = graph_result["messages"][-1]
            result = _completion_result(final, step.completion_criterion)
            if result is None:
                if isinstance(final, AIMessage) and final.tool_calls:
                    return StepExecution(revision, step_id, "failed", error="tool round budget exhausted")
                return StepExecution(revision, step_id, "failed", error="model did not return a valid completion result")
            return StepExecution(revision, step_id, "completed", result=result)
        except Exception as error:
            if self._checkpointer is not None:
                raise PersistenceError("checkpointed step execution stopped before further tool work") from error
            return StepExecution(revision, step_id, "failed", error=f"execution failed: {error}")

    async def _ensure_tools(self) -> None:
        if self._tools is not None:
            return
        self._client = MultiServerMCPClient({"atom": self._connection})
        self._session = self._client.session("atom")
        session = await self._session.__aenter__()
        tools = await load_mcp_tools(session)
        if self._tool_allowlist is not None:
            tools = [tool for tool in tools if tool.name in self._tool_allowlist]
        self._tools = tools
        self._bound_model = self._model.bind_tools(tools)
        tool_node = ToolNode(tools, handle_tool_errors=False)
        async def call_model(state: MessagesState) -> dict[str, list[BaseMessage]]:
            message = await self._bound_model.ainvoke(state["messages"])
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
            return END if any(_tool_error(message) is not None for message in _latest_tool_messages(state["messages"])) else "model"

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
            for message in result.get("messages", []):
                if isinstance(message, ToolMessage) and self._trace is not None:
                    self._trace.tool_result(
                        getattr(message, "name", "unknown"),
                        message.content,
                        error=_tool_error(message) is not None,
                    )
            return result

        graph.add_node("tools", call_tools)
        graph.add_edge(START, "mark_running")
        graph.add_edge("mark_running", "model")
        graph.add_conditional_edges("model", route, {"tools": "tools", END: END})
        graph.add_conditional_edges("tools", route_after_tools, {"model": "model", END: END})
        self._graph = graph.compile(checkpointer=self._checkpointer)

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("Executor must be used as an asynchronous context manager")


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
    if not isinstance(message.content, str):
        return None
    try:
        document = json.loads(message.content)
    except json.JSONDecodeError:
        return None
    if not isinstance(document, dict) or set(document) != {"completed", "result"}:
        return None
    if document["completed"] is not True or not isinstance(document["result"], str) or not document["result"].strip():
        return None
    result = document["result"].strip()
    criterion_statement = f"Completion criterion met: {completion_criterion}"
    explanation = result.removesuffix(criterion_statement).strip()
    if not result.endswith(criterion_statement) or not explanation:
        return None
    return result


def _latest_tool_messages(messages: list[BaseMessage]) -> list[ToolMessage]:
    tool_messages: list[ToolMessage] = []
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            break
        tool_messages.append(message)
    return tool_messages


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
