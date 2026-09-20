"""The public application seam for ordered, durable Conversations."""

from __future__ import annotations

import inspect
import asyncio
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Protocol

from .execution import ExecutionAnswer, ExecutionModeName, ToolRuntime, create_execution_mode
from .planner import PlanningValidationError
from .registry import ConfigurationMismatchError, MySQLRunRegistry, RunBusyError


ConversationStatus = Literal["pending", "running", "completed", "failed", "blocked"]


class ConversationError(RuntimeError):
    """Base class for Conversation application errors."""


class ConversationNotFoundError(ConversationError):
    pass


class ConversationBusyError(ConversationError):
    def __init__(self, conv_id: str, sequence: int, run_id: str) -> None:
        self.conv_id = conv_id
        self.sequence = sequence
        self.run_id = run_id
        super().__init__(f"conversation busy: turn {sequence} (run {run_id}) is active")


@dataclass(frozen=True, slots=True)
class ConversationInput:
    conv_id: str
    history: tuple[dict[str, Any], ...]
    current_input: str


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    conv_id: str
    sequence: int
    run_id: str
    execution_mode: str
    user_input: str
    status: ConversationStatus
    answer: str | None = None
    error: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "user_input": self.user_input,
            "execution_mode": self.execution_mode,
            "run_id": self.run_id,
            "status": self.status,
            "answer": self.answer,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class ConversationResult:
    conv_id: str
    sequence: int
    run_id: str
    execution_mode: str
    status: ConversationStatus
    answer: str | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "conv_id": self.conv_id,
            "sequence": self.sequence,
            "run_id": self.run_id,
            "execution_mode": self.execution_mode,
            "status": self.status,
            "answer": self.answer,
            "error": self.error,
        }


@dataclass
class Conversation:
    conv_id: str
    turns: list[ConversationTurn] = field(default_factory=list)


class ConversationStore(Protocol):
    def append(self, turn: ConversationTurn) -> Any: ...
    def get(self, conv_id: str) -> Conversation: ...
    def update(self, turn: ConversationTurn) -> Any: ...


class InMemoryConversationStore:
    """A deterministic store double and a useful local application store."""

    def __init__(self) -> None:
        self.conversations: dict[str, Conversation] = {}

    def append(self, turn: ConversationTurn) -> None:
        conversation = self.conversations.setdefault(turn.conv_id, Conversation(turn.conv_id))
        active = next((item for item in conversation.turns if item.status in {"pending", "running"}), None)
        if active is not None:
            raise ConversationBusyError(turn.conv_id, active.sequence, active.run_id)
        if turn.sequence != len(conversation.turns) + 1:
            raise ConversationError("conversation turn sequence conflict")
        conversation.turns.append(turn)

    def get(self, conv_id: str) -> Conversation:
        try:
            return self.conversations[conv_id]
        except KeyError as exc:
            raise ConversationNotFoundError(f"unknown conversation: {conv_id}") from exc

    def update(self, turn: ConversationTurn) -> None:
        conversation = self.get(turn.conv_id)
        for index, existing in enumerate(conversation.turns):
            if existing.sequence == turn.sequence:
                conversation.turns[index] = turn
                return
        raise ConversationError("unknown conversation turn")


class MySQLConversationStore:
    """Conversation persistence using the project-owned MySQL schema."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    async def append(self, turn: ConversationTurn) -> None:
        async with self._pool.acquire() as connection:
            begin = getattr(connection, "begin", None)
            if begin is not None:
                await begin()
            async with connection.cursor() as cursor:
                await cursor.execute("SELECT conv_id FROM conversations WHERE conv_id=%s FOR UPDATE", (turn.conv_id,))
                exists = await cursor.fetchone()
                if exists is None:
                    if turn.sequence != 1:
                        raise ConversationNotFoundError(f"unknown conversation: {turn.conv_id}")
                    await cursor.execute("INSERT INTO conversations (conv_id) VALUES (%s)", (turn.conv_id,))
                await cursor.execute(
                    "SELECT sequence, run_id FROM conversation_turns WHERE conv_id=%s AND status IN ('pending','running') FOR UPDATE",
                    (turn.conv_id,),
                )
                active = await cursor.fetchone()
                if active is not None:
                    raise ConversationBusyError(turn.conv_id, active[0], active[1])
                await cursor.execute(
                    "SELECT COALESCE(MAX(sequence), 0) FROM conversation_turns WHERE conv_id=%s",
                    (turn.conv_id,),
                )
                expected = (await cursor.fetchone())[0] + 1
                if turn.sequence != expected:
                    raise ConversationError("conversation turn sequence conflict")
                await cursor.execute(
                    "INSERT INTO conversation_turns (conv_id, sequence, run_id, execution_mode, user_input, status, answer, error) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    (turn.conv_id, turn.sequence, turn.run_id, turn.execution_mode, turn.user_input, turn.status, turn.answer, turn.error),
                )
            await connection.commit()

    async def get(self, conv_id: str) -> Conversation:
        async with self._pool.acquire() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute("SELECT conv_id FROM conversations WHERE conv_id=%s", (conv_id,))
                if await cursor.fetchone() is None:
                    raise ConversationNotFoundError(f"unknown conversation: {conv_id}")
                await cursor.execute(
                    "SELECT sequence, run_id, execution_mode, user_input, status, answer, error FROM conversation_turns WHERE conv_id=%s ORDER BY sequence",
                    (conv_id,),
                )
                rows = await cursor.fetchall()
        return Conversation(conv_id, [ConversationTurn(conv_id, *row) for row in rows])

    async def update(self, turn: ConversationTurn) -> None:
        async with self._pool.acquire() as connection:
            async with connection.cursor() as cursor:
                await cursor.execute(
                    "UPDATE conversation_turns SET status=%s, answer=%s, error=%s WHERE conv_id=%s AND sequence=%s",
                    (turn.status, turn.answer, turn.error, turn.conv_id, turn.sequence),
                )
                if cursor.rowcount != 1:
                    raise ConversationError("unknown conversation turn")
            await connection.commit()


class ConversationService:
    """Create, serialize, and expose Conversation turns through one seam."""

    def __init__(
        self,
        store: ConversationStore,
        runner_factory: Callable[..., Any],
        *,
        recovery_factory: Callable[..., Any] | None = None,
        run_result_factory: Callable[..., Any] | None = None,
        mode_selector: Callable[..., Any] | None = None,
    ) -> None:
        self._store = store
        self._runner_factory = runner_factory
        self._recovery_factory = recovery_factory or runner_factory
        self._run_result_factory = run_result_factory
        self._mode_selector = mode_selector
        self._active_ephemeral_runs: set[str] = set()

    async def run(
        self,
        *,
        user_input: str,
        execution_mode: ExecutionModeName = "direct",
        conv_id: str | None = None,
    ) -> ConversationResult:
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("conversation run requires nonempty user input")
        if execution_mode not in {"direct", "tool_agent", "react", "plan_execute"}:
            raise ValueError("execution mode must be direct, tool_agent, react, or plan_execute")
        if conv_id is None:
            conv_id = str(uuid.uuid4())
            history: tuple[dict[str, Any], ...] = ()
            sequence = 1
        else:
            _validate_uuid4(conv_id)
            conversation = await _maybe_await(self._store.get(conv_id))
            await self._reconcile_active(conversation)
            conversation = await _maybe_await(self._store.get(conv_id))
            history = tuple(turn.public() for turn in conversation.turns if turn.status not in {"pending", "running"})
            active = next((turn for turn in conversation.turns if turn.status in {"pending", "running"}), None)
            if active is not None:
                raise ConversationBusyError(conv_id, active.sequence, active.run_id)
            sequence = len(conversation.turns) + 1

        run_id = str(uuid.uuid4())
        conversation_input = ConversationInput(conv_id, history, user_input)
        if self._mode_selector is not None:
            execution_mode = await _maybe_await(
                _call_factory(self._mode_selector, conversation_input, run_id)
            )
            if execution_mode not in {"direct", "tool_agent", "react", "plan_execute"}:
                raise TypeError("task analyzer returned an invalid execution mode")
        turn = ConversationTurn(conv_id, sequence, run_id, execution_mode, user_input, "pending")
        await _maybe_await(self._store.append(turn))
        turn = ConversationTurn(conv_id, sequence, run_id, execution_mode, user_input, "running")
        await _maybe_await(self._store.update(turn))
        if execution_mode in {"direct", "tool_agent", "react"}:
            self._active_ephemeral_runs.add(run_id)

        runner = self._make_runner(execution_mode, run_id)
        try:
            answer = await runner.run(conversation_input)
            if not isinstance(answer, ExecutionAnswer):
                raise TypeError("execution mode returned an invalid answer")
        except asyncio.CancelledError:
            terminal = ConversationTurn(
                conv_id,
                sequence,
                run_id,
                execution_mode,
                user_input,
                "failed",
                error="conversation execution interrupted",
            )
            await _maybe_await(self._store.update(terminal))
            self._active_ephemeral_runs.discard(run_id)
            raise
        except (RunBusyError, ConfigurationMismatchError):
            raise
        except PlanningValidationError as error:
            answer = ExecutionAnswer(None, "failed", error=f"planning failed: {error}")
        except Exception:
            answer = ExecutionAnswer(None, "failed", error="conversation execution failed")
        self._active_ephemeral_runs.discard(run_id)
        terminal = ConversationTurn(
            conv_id, sequence, run_id, execution_mode, user_input, answer.status, answer.answer, answer.error
        )
        await _maybe_await(self._store.update(terminal))
        return ConversationResult(conv_id, sequence, run_id, execution_mode, answer.status, answer.answer, answer.error)

    async def resume(self, *, conv_id: str, recovery: str | None = None) -> ConversationResult:
        """Recover the one active durable turn and project its terminal answer."""
        _validate_uuid4(conv_id)
        conversation = await _maybe_await(self._store.get(conv_id))
        active = next((turn for turn in conversation.turns if turn.status in {"pending", "running"}), None)
        if active is None:
            raise ConversationError("conversation has no active turn")
        if active.execution_mode != "plan_execute":
            raise ConversationError("only a plan_execute turn can be resumed")
        runner = self._make_recovery_runner(active.run_id)
        try:
            answer = await runner.resume(recovery)
            if not isinstance(answer, ExecutionAnswer):
                raise TypeError("conversation recovery returned an invalid answer")
        except (RunBusyError, ConfigurationMismatchError):
            raise
        except PlanningValidationError as error:
            answer = ExecutionAnswer(None, "failed", error=f"planning failed: {error}")
        except Exception:
            answer = ExecutionAnswer(None, "failed", error="conversation recovery failed")
        terminal = ConversationTurn(
            conv_id, active.sequence, active.run_id, active.execution_mode, active.user_input,
            answer.status, answer.answer, answer.error,
        )
        await _maybe_await(self._store.update(terminal))
        return ConversationResult(
            conv_id, active.sequence, active.run_id, active.execution_mode,
            answer.status, answer.answer, answer.error,
        )

    async def show(self, conv_id: str) -> list[dict[str, Any]]:
        _validate_uuid4(conv_id)
        conversation = await _maybe_await(self._store.get(conv_id))
        return [turn.public() for turn in conversation.turns]

    def _make_runner(self, mode: str, run_id: str) -> Any:
        return _call_factory(self._runner_factory, mode, run_id)

    def _make_recovery_runner(self, run_id: str) -> Any:
        return _call_factory(self._recovery_factory, run_id)

    async def _reconcile_active(self, conversation: Conversation) -> None:
        active = next((turn for turn in conversation.turns if turn.status in {"pending", "running"}), None)
        if active is None:
            return
        if active.execution_mode in {"direct", "tool_agent", "react"}:
            if active.run_id not in self._active_ephemeral_runs:
                await _maybe_await(
                    self._store.update(
                        ConversationTurn(
                            active.conv_id,
                            active.sequence,
                            active.run_id,
                            active.execution_mode,
                            active.user_input,
                            "failed",
                            error="ephemeral conversation turn could not be recovered",
                        )
                    )
                )
            return
        if self._run_result_factory is None:
            return
        result = await _maybe_await(self._run_result_factory(active.run_id))
        if not isinstance(result, Mapping):
            return
        status = result.get("status")
        if status not in {"completed", "failed", "blocked"}:
            return
        terminal = ConversationTurn(
            active.conv_id, active.sequence, active.run_id, active.execution_mode, active.user_input,
            status, result.get("answer"), result.get("error"),
        )
        await _maybe_await(self._store.update(terminal))


def run_mysql_conversation(
    url: str,
    *,
    user_input: str,
    conv_id: str | None,
    configuration: Any,
    cwd: str | None = None,
    log_level: str = "info",
) -> ConversationResult:
    return asyncio.run(
        _run_mysql_conversation(
            url,
            user_input=user_input,
            conv_id=conv_id,
            configuration=configuration,
            cwd=cwd,
            log_level=log_level,
        )
    )


def show_mysql_conversation(url: str, conv_id: str) -> list[dict[str, Any]]:
    return asyncio.run(_show_mysql_conversation(url, conv_id))


def resume_mysql_conversation(
    url: str,
    *,
    conv_id: str,
    recovery: str | None,
    configuration: Any,
) -> ConversationResult:
    return asyncio.run(
        _resume_mysql_conversation(
            url, conv_id=conv_id, recovery=recovery, configuration=configuration
        )
    )


async def _run_mysql_conversation(url: str, **kwargs: Any) -> ConversationResult:
    from .durable import DurableConversationRunner
    from .migration import _connection_pool, _parse_url, ensure_schema_initialized

    def runner_factory(mode: str, run_id: str) -> Any:
        if mode == "plan_execute":
            return DurableConversationRunner(
                _MySQLDurableAgent(
                    url,
                    kwargs["configuration"],
                    kwargs.get("cwd"),
                    log_level=kwargs.get("log_level", "info"),
                ),
                run_id,
            )
        return _make_ephemeral_runner(
            mode,
            run_id,
            configuration=kwargs["configuration"],
            cwd=kwargs.get("cwd"),
            log_level=kwargs.get("log_level", "info"),
        )

    async def select_mode(conversation_input: ConversationInput, run_id: str) -> ExecutionModeName:
        return await _select_conversation_mode(
            conversation_input,
            run_id,
            configuration=kwargs["configuration"],
            log_level=kwargs.get("log_level", "info"),
        )

    async with _connection_pool(_parse_url(url)) as pool:
        await ensure_schema_initialized(pool)
        store = MySQLConversationStore(pool)
        service = ConversationService(
            store,
            runner_factory,
            run_result_factory=lambda run_id: _mysql_run_result(pool, run_id),
            mode_selector=select_mode,
        )
        return await service.run(
            user_input=kwargs["user_input"],
            conv_id=kwargs["conv_id"],
        )


async def _select_conversation_mode(
    conversation_input: ConversationInput,
    run_id: str,
    *,
    configuration: Any,
    log_level: str,
) -> ExecutionModeName:
    from llm import LLM
    from .task_analyzer import TaskAnalyzer, route
    from .trace import RunTrace

    trace = RunTrace(level=log_level, run_id=run_id)
    analyzer_configuration = configuration.task_analyzer
    analyzer_llm = LLM(
        analyzer_configuration.base_url,
        analyzer_configuration.api_key,
        analyzer_configuration.model_name,
        response_format=analyzer_configuration.response_format,
        on_event=trace.llm_event,
        on_request=lambda request: trace.llm_request("task_analyzer", request),
    )
    try:
        mode = route(
            await TaskAnalyzer(
                analyzer_llm,
                response_format=analyzer_configuration.response_format,
            ).run(conversation_input.current_input)
        )
        trace.task_route(mode)
        return mode
    except Exception:
        trace.task_route("plan_execute", analysis_failed=True)
        return "plan_execute"
    finally:
        await analyzer_llm.close()


def _make_ephemeral_runner(
    mode: str,
    run_id: str,
    *,
    configuration: Any,
    cwd: str | None,
    log_level: str,
) -> Any:
    """Create an ephemeral mode with the same host-selected tool boundary as durable runs."""
    from .trace import RunTrace

    tool_cwd = os.path.abspath(os.path.expanduser(cwd or os.getcwd()))
    trace = RunTrace(level=log_level, run_id=run_id)
    return create_execution_mode(
        mode,
        configuration=configuration,
        run_id=run_id,
        trace=trace,
        tool_runtime=ToolRuntime(tool_cwd=tool_cwd, trace=trace),
    )


async def _resume_mysql_conversation(
    url: str,
    *,
    conv_id: str,
    recovery: str | None,
    configuration: Any,
) -> ConversationResult:
    from .migration import _connection_pool, _parse_url, ensure_schema_initialized

    async with _connection_pool(_parse_url(url)) as pool:
        await ensure_schema_initialized(pool)
        def runner_factory(mode: str, run_id: str) -> Any:
            if mode != "plan_execute":
                return None
            return DurableConversationRunner(
                _MySQLDurableAgent(url, configuration, None, restore_cwd=True),
                run_id,
            )

        service = ConversationService(
            MySQLConversationStore(pool),
            runner_factory,
            recovery_factory=lambda run_id: runner_factory("plan_execute", run_id),
        )
        return await service.resume(conv_id=conv_id, recovery=recovery)


class _MySQLDurableAgent:
    """Small adapter that keeps Conversation recovery on the Agent seam."""

    def __init__(
        self,
        url: str,
        configuration: Any,
        cwd: str | None,
        *,
        log_level: str = "info",
        restore_cwd: bool = False,
    ) -> None:
        self._url = url
        self._configuration = configuration
        self._cwd = cwd
        self._log_level = log_level
        self._restore_cwd = restore_cwd

    async def run(
        self,
        run_id: str,
        state: Any = None,
        recovery: str | None = None,
        conversation_input: Any = None,
    ) -> dict[str, Any]:
        from .durable import _run

        goal = None if state is None else state.goal
        cwd = self._cwd
        if state is None and self._restore_cwd:
            from .migration import _connection_pool, _parse_url

            async with _connection_pool(_parse_url(self._url)) as pool:
                record = await MySQLRunRegistry(pool).get(run_id)
            cwd = record.config_snapshot.get("mcp", {}).get("tool_cwd")
        return await _run(
            self._url,
            run_id,
            goal,
            recovery=recovery,
            cwd=cwd,
            log_level=self._log_level,
            configuration=self._configuration,
            execution_mode="plan_execute",
            conversation_input=conversation_input,
        )


async def _mysql_run_result(pool: Any, run_id: str) -> Mapping[str, Any] | None:
    from .registry import MySQLRunRegistry

    record = await MySQLRunRegistry(pool).get(run_id)
    return record.terminal_result or {"status": record.status}


async def _show_mysql_conversation(url: str, conv_id: str) -> list[dict[str, Any]]:
    from .migration import _connection_pool, _parse_url, ensure_schema_initialized

    _validate_uuid4(conv_id)
    async with _connection_pool(_parse_url(url)) as pool:
        await ensure_schema_initialized(pool)
        return await ConversationService(MySQLConversationStore(pool), lambda mode, run_id: None).show(conv_id)


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _call_factory(factory: Callable[..., Any], *args: Any) -> Any:
    try:
        return factory(*args)
    except TypeError as error:
        if "positional" not in str(error) and "required" not in str(error):
            raise
        return factory(args[-1])


def _validate_uuid4(value: str) -> None:
    try:
        if uuid.UUID(value).version != 4:
            raise ValueError
    except (ValueError, AttributeError) as exc:
        raise ValueError("conversation requires a UUIDv4 conv_id") from exc
