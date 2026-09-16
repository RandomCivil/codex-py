"""Checkpointed top-level Agent graph and its CLI-facing lifecycle."""

import asyncio
import os
import signal
import uuid
from contextlib import asynccontextmanager, contextmanager, suppress
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from llm import LLM
from memory.state import (
    AgentState,
    StepExecution,
    deserialize_agent_state,
    serialize_agent_state,
)

from .agent import RecoveryDecisionError, apply_recovery_decision, mark_stale_execution_interrupted
from .executor import Executor
from .migration import _connection_pool, _parse_url, ensure_schema_initialized
from .planner import Planner
from .registry import (
    ConfigurationMismatchError,
    MySQLRunRegistry,
    RunBusyError,
    configuration_snapshot,
)


class GraphState(TypedDict, total=False):
    agent_state: dict[str, Any]
    status: str


def new_run_id() -> str:
    return str(uuid.uuid4())


class DurableAgent:
    """Run the existing Planner and Executor behind a durable LangGraph graph."""

    def __init__(self, planner: Planner, executor: Executor, saver: Any, trace: Any | None = None) -> None:
        self._planner = planner
        self._executor = executor
        self._saver = saver
        self._trace = trace

    def _graph(self):
        graph = StateGraph(GraphState)

        async def plan(state: GraphState) -> GraphState:
            agent_state = deserialize_agent_state(state["agent_state"])
            if agent_state.plan_history:
                return {}
            return {"agent_state": serialize_agent_state(agent_state.with_plan(await self._planner.plan(agent_state)))}

        async def mark_running(state: GraphState) -> GraphState:
            agent_state = deserialize_agent_state(state["agent_state"])
            step = _next_step(agent_state)
            if step is None:
                return {"status": "completed"}
            revision, step_id = step
            current = next(
                (item for item in agent_state.step_executions if (item.revision, item.step_id) == (revision, step_id)),
                None,
            )
            if current is not None and current.status == "failed":
                return {}
            updated = agent_state.with_step_execution(StepExecution(revision, step_id, "running"))
            return {"agent_state": serialize_agent_state(updated)}

        async def execute(state: GraphState) -> GraphState:
            agent_state = deserialize_agent_state(state["agent_state"])
            revision, step_id = _next_step(agent_state)
            if revision is None:
                return {"status": "completed"}
            if self._trace is not None:
                self._trace.execute("started", revision, step_id)
            async with self._executor as executor:
                execution = await executor.execute(agent_state, revision, step_id)
            if self._trace is not None:
                self._trace.execute(execution.status, revision, step_id, execution.error or execution.result)
            return {"agent_state": serialize_agent_state(agent_state.with_step_execution(execution))}

        async def replan(state: GraphState) -> GraphState:
            agent_state = deserialize_agent_state(state["agent_state"])
            if agent_state.plan_history[-1].revision == 3:
                if self._trace is not None:
                    self._trace.plan("blocked", 3)
                return {"status": "blocked"}
            if self._trace is not None:
                self._trace.plan("replanning", len(agent_state.plan_history) + 1)
            return {"agent_state": serialize_agent_state(agent_state.with_plan(await self._planner.plan(agent_state)))}

        async def terminal(state: GraphState) -> GraphState:
            return {"status": "completed"}

        graph.add_node("plan", plan)
        graph.add_node("mark_running", mark_running)
        graph.add_node("execute", execute)
        graph.add_node("replan", replan)
        graph.add_node("complete", terminal)
        graph.add_node("blocked", lambda state: {"status": "blocked"})
        graph.add_node("interrupted", lambda state: {"status": "interrupted"})
        graph.add_conditional_edges(
            START,
            _start_node,
            {"plan": "plan", "blocked": "blocked", "interrupted": "interrupted"},
        )
        graph.add_conditional_edges("plan", _after_plan, {"mark_running": "mark_running", "complete": "complete"})
        graph.add_conditional_edges(
            "mark_running",
            _after_mark_running,
            {"execute": "execute", "replan": "replan", "complete": "complete"},
        )
        graph.add_conditional_edges("execute", _after_execute, {"execute": "mark_running", "replan": "replan", "complete": "complete"})
        graph.add_conditional_edges("replan", _after_replan, {"mark_running": "mark_running", "blocked": END})
        graph.add_edge("complete", END)
        graph.add_edge("blocked", END)
        graph.add_edge("interrupted", END)
        return graph.compile(checkpointer=self._saver)

    async def run(
        self,
        run_id: str,
        state: AgentState | None = None,
        recovery: str | None = None,
    ) -> dict[str, Any]:
        config = {"configurable": {"thread_id": run_id}}
        graph = self._graph()
        if state is None:
            checkpoint = await graph.aget_state(config)
            values = checkpoint.values
            if values.get("agent_state"):
                state = deserialize_agent_state(values["agent_state"])
                if values.get("status") in {"completed", "blocked"}:
                    return {
                        "run_id": run_id,
                        "status": values["status"],
                        "state": values["agent_state"],
                    }
        if state is not None:
            state, abort = apply_recovery_decision(state, recovery)
            if abort:
                result = await graph.ainvoke(
                    {"agent_state": serialize_agent_state(state), "status": "blocked"}, config=config
                )
                return {"run_id": run_id, "status": "blocked", "state": result["agent_state"]}
        result = await graph.ainvoke(
            {"agent_state": serialize_agent_state(state), "status": "resuming"} if state is not None else None,
            config=config,
        )
        return {"run_id": run_id, "status": result.get("status", "completed"), "state": result.get("agent_state")}

    async def interrupt(self, run_id: str) -> None:
        """Durably mark in-progress work uncertain after cancellation."""
        config = {"configurable": {"thread_id": run_id}}
        graph = self._graph()
        checkpoint = await graph.aget_state(config)
        state_payload = checkpoint.values.get("agent_state")
        if not state_payload:
            return
        state = mark_stale_execution_interrupted(deserialize_agent_state(state_payload))
        await graph.ainvoke(
            {"agent_state": serialize_agent_state(state), "status": "interrupted"}, config=config
        )


@asynccontextmanager
async def _mysql_saver(url: str):
    from langgraph.checkpoint.mysql.aio import AIOMySQLSaver

    async with AIOMySQLSaver.from_conn_string(url) as saver:
        yield saver


async def _run(
    url: str,
    run_id: str,
    goal: str | None,
    recovery: str | None = None,
    cwd: str | None = None,
) -> dict[str, Any]:
    model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    tool_cwd = os.path.abspath(os.path.expanduser(cwd or os.getcwd()))
    snapshot = configuration_snapshot(
        base_url=os.environ.get("OPENAI_BASE_URL", ""),
        model=model_name,
        mcp={
            "command": "poetry",
            "args": ["run", "atom-mcp"],
            "cwd": "/home/xzp/workspace/atom-mcp",
            "tool_cwd": tool_cwd,
        },
    )
    owner = str(uuid.uuid4())
    async with _connection_pool(_parse_url(url)) as pool:
        await ensure_schema_initialized(pool)
        registry = MySQLRunRegistry(pool)
        if goal is not None:
            await registry.create(run_id, snapshot)
        else:
            existing = await registry.ensure_compatible(run_id, snapshot)
            if existing.status in {"completed", "blocked"} and existing.terminal_result is not None:
                return dict(existing.terminal_result)
        await registry.acquire(run_id, owner)
        renewal = asyncio.create_task(_renew_lease(registry, run_id, owner))
        try:
            async with _mysql_saver(url) as saver:
                try:
                    from .trace import RunTrace

                    trace = RunTrace()
                    llm = LLM(
                        os.environ.get("OPENAI_BASE_URL", ""),
                        os.environ.get("OPENAI_API_KEY", ""),
                        model_name,
                        on_event=trace.llm_event,
                    )
                    planner = Planner(llm, trace=trace)
                    executor = Executor(
                        model_name=model_name,
                        checkpointer=saver,
                        run_id=run_id,
                        trace=trace,
                        tool_cwd=tool_cwd,
                    )
                    initial = AgentState(goal) if goal is not None else None
                    durable = DurableAgent(planner, executor, saver, trace=trace)
                    with _cancel_on_signals():
                        result = await durable.run(run_id, initial, recovery)
                except (asyncio.CancelledError, KeyboardInterrupt):
                    try:
                        await asyncio.wait_for(durable.interrupt(run_id), timeout=5)
                    except Exception:
                        pass
                    raise
                finally:
                    await llm.close()
            if result["status"] in {"completed", "blocked"}:
                await registry.set_terminal(run_id, owner, {"run_id": run_id, "status": result["status"]})
            return result
        finally:
            renewal.cancel()
            with suppress(asyncio.CancelledError):
                await renewal
            await registry.release(run_id, owner)


def run_agent(url: str, goal: str, cwd: str | None = None) -> dict[str, Any]:
    return asyncio.run(_run(url, new_run_id(), goal, cwd=cwd))


def resume_agent(url: str, run_id: str, recovery: str | None = None, cwd: str | None = None) -> dict[str, Any]:
    return asyncio.run(_run(url, run_id, None, recovery, cwd))


async def _renew_lease(registry: Any, run_id: str, owner: str) -> None:
    while True:
        await asyncio.sleep(15)
        await registry.renew(run_id, owner)


@contextmanager
def _cancel_on_signals():
    """Turn ordinary process shutdown signals into cancellable durable work."""
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    if task is None:
        yield
        return
    installed: list[tuple[signal.Signals, Any]] = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous = signal.getsignal(signum)
            loop.add_signal_handler(signum, task.cancel)
            installed.append((signum, previous))
        except (NotImplementedError, RuntimeError):
            continue
    try:
        yield
    finally:
        for signum, previous in installed:
            loop.remove_signal_handler(signum)
            signal.signal(signum, previous)


def _next_step(state: AgentState) -> tuple[int | None, str | None]:
    if not state.plan_history:
        return None, None
    plan = state.plan_history[-1]
    for step in plan.steps:
        execution = next((item for item in state.step_executions if (item.revision, item.step_id) == (plan.revision, step.id)), None)
        if execution is None or execution.status != "completed":
            return plan.revision, step.id
    return None, None


def _after_plan(state: GraphState) -> str:
    return "complete" if _next_step(deserialize_agent_state(state["agent_state"])) == (None, None) else "mark_running"


def _after_mark_running(state: GraphState) -> str:
    agent_state = deserialize_agent_state(state["agent_state"])
    if agent_state.step_executions and agent_state.step_executions[-1].status == "failed":
        return "replan"
    return "complete" if _next_step(agent_state) == (None, None) else "execute"


def _after_execute(state: GraphState) -> str:
    agent_state = deserialize_agent_state(state["agent_state"])
    execution = agent_state.step_executions[-1]
    if execution.status == "failed":
        return "replan"
    return "complete" if _next_step(agent_state) == (None, None) else "execute"


def _after_replan(state: GraphState) -> str:
    return "blocked" if state.get("status") == "blocked" else "mark_running"


def _start_node(state: GraphState) -> str:
    if state.get("status") in {"blocked", "interrupted"}:
        return state["status"]
    return "plan"
