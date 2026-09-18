"""Checkpointed top-level Agent graph and its CLI-facing lifecycle."""

import asyncio
import os
import signal
import uuid
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import asdict
from typing import Any, TypedDict
from langgraph.graph import END, START, StateGraph

from llm import LLM
from memory.state import (
    AgentState,
    ContextUpdate,
    ExecutionOutcome,
    StepExecution,
    StepContext,
    deserialize_agent_state,
    serialize_agent_state,
)

from .agent import RecoveryDecisionError, apply_recovery_decision, mark_stale_execution_interrupted
from .configuration import ComponentProviderConfiguration
from .executor import Executor
from .migration import _connection_pool, _parse_url, ensure_schema_initialized
from .planner import Planner
from .execution import ToolRuntime, create_execution_mode
from .task_analyzer import RoutedExecutionAnswer, TaskAnalyzer, TaskRouter
from .registry import (
    ConfigurationMismatchError,
    MySQLRunRegistry,
    RunBusyError,
    configuration_snapshot,
)
from .recovery import MySQLStepRecoveryStore


class GraphState(TypedDict, total=False):
    agent_state: AgentState
    status: str
    run_id: str
    # 发生变化的文件路径
    files_modified: list[str]
    # 读取过的文件路径
    files_read: list[str]
    # 对读取过的文件的总结出的关键发现
    observations: list[str]
    persistence_version: int
    recovery_attempt: int
    recovery_step: tuple[int, str]
    recovery_active: bool


def new_run_id() -> str:
    return str(uuid.uuid4())


class DurableAgent:
    """Run the existing Planner and Executor behind a durable LangGraph graph."""

    def __init__(
        self,
        planner: Planner,
        executor: Executor,
        saver: Any,
        trace: Any | None = None,
        recovery_store: MySQLStepRecoveryStore | None = None,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._saver = saver
        self._trace = trace
        self._recovery_store = recovery_store

    def _graph(self):
        graph = StateGraph(GraphState)

        async def plan(state: GraphState) -> GraphState:
            agent_state = state["agent_state"]
            if agent_state.plan_history:
                return {}
            return {"agent_state": agent_state.with_plan(await self._planner.plan(agent_state))}

        async def mark_running(state: GraphState) -> GraphState:
            agent_state = state["agent_state"]
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
            version = state.get("persistence_version", 0) + 1
            if self._recovery_store is not None:
                await self._recovery_store.record(
                    state.get("run_id", ""),
                    updated.step_executions[-1],
                    version=version,
                )
            return {
                "agent_state": updated,
                "persistence_version": version,
                "recovery_active": state.get("recovery_active", False),
            }

        async def execute(state: GraphState) -> GraphState:
            agent_state = state["agent_state"]
            revision, step_id = _next_step(agent_state)
            if revision is None:
                return {"status": "completed"}
            if self._trace is not None:
                self._trace.execute("started", revision, step_id)
            async with self._executor as executor:
                outcome = await executor.execute(
                    agent_state,
                    revision,
                    step_id,
                    StepContext(
                        files_read=tuple(state.get("files_read", [])),
                        files_modified=tuple(state.get("files_modified", [])),
                        observations=tuple(state.get("observations", [])),
                    ),
                    recovery=state.get("recovery_active", False),
                    attempt=state.get("recovery_attempt"),
                )
                execution = outcome.execution
            if self._trace is not None:
                self._trace.execute(execution.status, revision, step_id, execution.error or execution.result)
            updates: GraphState = {"agent_state": agent_state.with_step_execution(execution)}
            if isinstance(outcome, ExecutionOutcome) and outcome.context_update is not None:
                updates.update(_merge_context(state, outcome.context_update, revision, step_id))
            version = state.get("persistence_version", 0) + 1
            updates["persistence_version"] = version
            updates["recovery_active"] = False
            if self._recovery_store is not None:
                await self._recovery_store.record(
                    state.get("run_id", ""),
                    execution,
                    context_update=outcome.context_update if isinstance(outcome, ExecutionOutcome) else None,
                    recovery=state.get("status") == "resuming",
                    version=version,
                )
            return updates

        async def replan(state: GraphState) -> GraphState:
            agent_state = state["agent_state"]
            if agent_state.plan_history[-1].revision == 3:
                if self._trace is not None:
                    self._trace.plan("blocked", 3)
                return {"status": "blocked"}
            if self._trace is not None:
                self._trace.plan("replanning", len(agent_state.plan_history) + 1)
            return {"agent_state": agent_state.with_plan(await self._planner.plan(agent_state))}

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
        recovery_attempt: int | None = None
        recovery_active = False
        restored_context: GraphState = {}
        if state is None:
            checkpoint = await graph.aget_state(config)
            values = checkpoint.values
            if self._recovery_store is not None:
                snapshot = await self._recovery_store.restore(
                    run_id, values.get("persistence_version", 0)
                )
                if snapshot.latest_executions and values.get("agent_state"):
                    restored = values["agent_state"]
                    if isinstance(restored, dict):
                        restored = deserialize_agent_state(restored)
                    state = _restore_recovery_state(restored, snapshot.latest_executions)
                    values = dict(values)
                    values["agent_state"] = state
                    values["files_read"] = list(snapshot.context.files_read)
                    values["files_modified"] = list(snapshot.context.files_modified)
                    values["observations"] = list(snapshot.context.observations)
                    restored_context = {
                        "files_read": values["files_read"],
                        "files_modified": values["files_modified"],
                        "observations": values["observations"],
                    }
                    interrupted = next(
                        (item for item in snapshot.latest_executions if item.status == "interrupted"),
                        None,
                    )
                    if interrupted is not None:
                        recovery_active = True
                        prior = [
                            item for item in snapshot.attempts
                            if (item.revision, item.step_id) == (interrupted.revision, interrupted.step_id)
                        ]
                        recovery_attempt = (prior[-1].attempt + 1) if prior else 1
            if values.get("agent_state"):
                state = values["agent_state"]
                if isinstance(state, dict):
                    state = deserialize_agent_state(state)
                if self._trace is not None:
                    self._trace.recovery_context(
                        list(values.get("files_read", [])),
                        list(values.get("files_modified", [])),
                        list(values.get("observations", [])),
                    )
                if values.get("status") in {"completed", "blocked"}:
                    return {
                        "run_id": run_id,
                        "status": values["status"],
                        "state": serialize_agent_state(state),
                    }
        if state is not None:
            abort = False
            stale = mark_stale_execution_interrupted(state)
            interrupted = next((item for item in stale.step_executions if item.status == "interrupted"), None)
            was_running = any(item.status == "running" for item in state.step_executions)
            if interrupted is not None and recovery is None:
                state = stale
                recovery_active = True
                if self._recovery_store is not None:
                    history = await self._recovery_store.history(run_id)
                    prior = [
                        item for item in history
                        if (item.revision, item.step_id) == (interrupted.revision, interrupted.step_id)
                    ]
                    recovery_attempt = (prior[-1].attempt + 1) if prior else 1
            else:
                state, abort = apply_recovery_decision(stale, recovery)
            if was_running and interrupted is not None and self._recovery_store is not None:
                history = await self._recovery_store.history(run_id)
                version = (history[-1].version + 1) if history else 1
                interrupted_record = await self._recovery_store.record(run_id, interrupted, version=version)
                recovery_active = recovery is None
                recovery_attempt = interrupted_record.attempt + 1
            if abort:
                result = await graph.ainvoke(
                    {"agent_state": state, "status": "blocked"}, config=config
                )
                return {"run_id": run_id, "status": "blocked", "state": serialize_agent_state(result["agent_state"])}
        result = await graph.ainvoke(
            ({
                "agent_state": state,
                "status": "resuming",
                "run_id": run_id,
                "recovery_active": recovery_active,
                **restored_context,
                **({"recovery_attempt": recovery_attempt} if recovery_attempt is not None else {}),
            } if state is not None else {"run_id": run_id}),
            config=config,
        )
        return {
            "run_id": run_id,
            "status": result.get("status", "completed"),
            "state": serialize_agent_state(result["agent_state"]) if result.get("agent_state") else None,
        }

    async def interrupt(self, run_id: str) -> None:
        """Durably mark in-progress work uncertain after cancellation."""
        config = {"configurable": {"thread_id": run_id}}
        graph = self._graph()
        checkpoint = await graph.aget_state(config)
        state_payload = checkpoint.values.get("agent_state")
        if not state_payload:
            return
        if isinstance(state_payload, dict):
            state_payload = deserialize_agent_state(state_payload)
        was_running = any(item.status == "running" for item in state_payload.step_executions)
        state = mark_stale_execution_interrupted(state_payload)
        version = checkpoint.values.get("persistence_version", 0) + 1
        if was_running and self._recovery_store is not None:
            interrupted = next(item for item in state.step_executions if item.status == "interrupted")
            await self._recovery_store.record(run_id, interrupted, version=version)
        await graph.ainvoke(
            {
                "agent_state": state,
                "status": "interrupted",
                "run_id": run_id,
                "persistence_version": version,
            },
            config=config,
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
    log_level: str = "info",
    configuration: ComponentProviderConfiguration | None = None,
) -> dict[str, Any]:
    if configuration is None:
        raise ValueError("an explicit component provider configuration is required")
    planner_configuration = configuration.planner
    executor_configuration = configuration.executor
    analyzer_configuration = configuration.task_analyzer
    tool_cwd = os.path.abspath(os.path.expanduser(cwd or os.getcwd()))
    snapshot = configuration_snapshot(
        planner=planner_configuration.__dict__,
        executor=executor_configuration.__dict__,
        task_analyzer=analyzer_configuration.__dict__,
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
            if existing.status in {"completed", "blocked", "failed"} and existing.terminal_result is not None:
                return dict(existing.terminal_result)
        await registry.acquire(run_id, owner)
        renewal = asyncio.create_task(_renew_lease(registry, run_id, owner))
        try:
            async with _mysql_saver(url) as saver:
                try:
                    from .trace import RunTrace

                    trace = RunTrace(level=log_level, run_id=run_id)
                    llm = LLM(
                        planner_configuration.base_url,
                        planner_configuration.api_key,
                        planner_configuration.model_name,
                        response_format=planner_configuration.response_format,
                        on_event=trace.llm_event,
                    )
                    planner = Planner(llm, trace=trace, response_format=planner_configuration.response_format)
                    executor = Executor(
                        base_url=executor_configuration.base_url,
                        api_key=executor_configuration.api_key,
                        model_name=executor_configuration.model_name,
                        response_format=executor_configuration.response_format,
                        checkpointer=saver,
                        run_id=run_id,
                        trace=trace,
                        tool_cwd=tool_cwd,
                    )
                    initial = AgentState(goal) if goal is not None else None
                    durable = DurableAgent(
                        planner,
                        executor,
                        saver,
                        trace=trace,
                        recovery_store=MySQLStepRecoveryStore(pool),
                    )
                    with _cancel_on_signals():
                        if goal is None:
                            result = await durable.run(run_id, initial, recovery)
                        else:
                            analyzer_llm = LLM(
                                analyzer_configuration.base_url,
                                analyzer_configuration.api_key,
                                analyzer_configuration.model_name,
                                response_format=analyzer_configuration.response_format,
                                on_event=trace.llm_event,
                            )
                            try:
                                router = _task_router(
                                    TaskAnalyzer(
                                        analyzer_llm,
                                        response_format=analyzer_configuration.response_format,
                                    ),
                                    configuration=configuration,
                                    durable_agent=durable,
                                    tool_cwd=tool_cwd,
                                    trace=trace,
                                )
                                result = _routed_result(run_id, await router.run(goal))
                            finally:
                                await analyzer_llm.close()
                except (asyncio.CancelledError, KeyboardInterrupt):
                    try:
                        await asyncio.wait_for(durable.interrupt(run_id), timeout=5)
                    except Exception:
                        pass
                    raise
                finally:
                    await llm.close()
            if result["status"] in {"completed", "blocked", "failed"}:
                await registry.set_terminal(run_id, owner, result)
            return result
        finally:
            renewal.cancel()
            with suppress(asyncio.CancelledError):
                await renewal
            await registry.release(run_id, owner)


def run_agent(
    url: str,
    goal: str,
    cwd: str | None = None,
    log_level: str = "info",
    *,
    configuration: ComponentProviderConfiguration | None = None,
) -> dict[str, Any]:
    return asyncio.run(_run(url, new_run_id(), goal, cwd=cwd, log_level=log_level, configuration=configuration))


def resume_agent(
    url: str,
    run_id: str,
    recovery: str | None = None,
    cwd: str | None = None,
    log_level: str = "info",
    *,
    configuration: ComponentProviderConfiguration | None = None,
) -> dict[str, Any]:
    return asyncio.run(_run(url, run_id, None, recovery, cwd, log_level, configuration))


def _routed_result(run_id: str, routed: RoutedExecutionAnswer) -> dict[str, Any]:
    """Serialize the Task Router outcome for the CLI-facing run lifecycle."""
    return {
        "run_id": run_id,
        "status": routed.execution.status,
        "execution_mode": routed.execution_mode,
        "execution": asdict(routed.execution),
        "analysis": asdict(routed.analysis) if routed.analysis is not None else None,
        "analysis_error": routed.analysis_error,
    }


def _task_router(
    analyzer: TaskAnalyzer,
    *,
    configuration: ComponentProviderConfiguration,
    durable_agent: DurableAgent,
    tool_cwd: str,
    trace: Any,
) -> TaskRouter:
    """Compose the one run-entry router with the existing mode-factory seam."""
    return TaskRouter(
        analyzer,
        lambda mode: create_execution_mode(
            mode,
            configuration=configuration,
            durable_agent=durable_agent,
            tool_runtime=ToolRuntime(tool_cwd=tool_cwd, trace=trace),
            trace=trace,
        ),
    )


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


def _restore_recovery_state(
    state: AgentState,
    executions: tuple[StepExecution, ...],
) -> AgentState:
    """Overlay the latest project recovery projection on checkpointed plans."""
    restored = state
    for execution in executions:
        restored = restored.with_step_execution(execution)
    return restored


def _after_plan(state: GraphState) -> str:
    return "complete" if _next_step(state["agent_state"]) == (None, None) else "mark_running"


def _after_mark_running(state: GraphState) -> str:
    agent_state = state["agent_state"]
    if agent_state.step_executions and agent_state.step_executions[-1].status == "failed":
        return "replan"
    return "complete" if _next_step(agent_state) == (None, None) else "execute"


def _after_execute(state: GraphState) -> str:
    agent_state = state["agent_state"]
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


def _merge_context(
    state: GraphState,
    update: ContextUpdate,
    revision: int,
    step_id: str,
) -> GraphState:
    """Merge one successful update into cumulative checkpointed Step context."""

    def append_unique(existing: list[str], values: tuple[str, ...]) -> list[str]:
        merged = list(existing)
        seen = set(merged)
        for value in values:
            if value not in seen:
                merged.append(value)
                seen.add(value)
        return merged

    observations = [
        *state.get("observations", []),
        *(
            f"revision {revision}, step {step_id}: {observation}"
            for observation in update.observations
        ),
    ]
    return {
        "files_read": append_unique(state.get("files_read", []), update.files_read),
        "files_modified": append_unique(state.get("files_modified", []), update.files_modified),
        "observations": observations,
    }
