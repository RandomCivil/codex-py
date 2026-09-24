"""Layered Runtime context for tool-capable model loops.

The policy deliberately keeps transient tool evidence separate from durable
state.  Callers may use :meth:`assemble` for the current in-memory window and
``maintain`` when budget-driven Observation merging is needed.
"""

from __future__ import annotations

import asyncio
import json
import shlex
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from langchain_core.messages import HumanMessage

from llm.line_protocol import LineProtocolError, parse_line_protocol


DEFAULT_CONTEXT_BUDGET = 128_000
RAW_ROUND_WINDOW = 3
NATIVE_LISTING_ROUND_WINDOW = 1

EvidenceLifecycle = Literal["current_raw", "recent_raw", "permanent_raw", "observation"]


_OBSERVATION_CONTRACT = """Return exactly one UTF-8 Line Protocol block:
BEGIN OBSERVATION ... END OBSERVATION.
Create the Observation for the settled Tool round in the next message.
Assess whether the evidence changes the active decision. Set
AFFECTS_CURRENT_DECISION to true or false and repeat AFFECTED_TARGETS for any of
confirmed_facts, summary, or durable_state affected. Positive impact requires at
least one target; negative impact requires none.
Set ROUND to the source round. Every EVIDENCE block MUST contain CATEGORY, TEXT, and
at least one TOOL_CALL_ID. Copy TOOL_CALL_ID exactly from the input tool-call ID;
never omit it and never invent one. Classify tool-confirmed output as confirmed_facts,
every raw error as reported_errors, and interpretations as model_inferences. Use JSON
literals after '=' and do not output prose.

Example:
BEGIN OBSERVATION
ROUND=4
AFFECTS_CURRENT_DECISION=true
AFFECTED_TARGETS="confirmed_facts"
BEGIN EVIDENCE
CATEGORY="confirmed_facts"
TEXT="The requested files were found"
TOOL_CALL_ID="call_123"
END EVIDENCE
END OBSERVATION"""

_MERGE_OBSERVATION_CONTRACT = """Return exactly one UTF-8 Line Protocol block:
BEGIN OBSERVATION ... END OBSERVATION.
Merge the Observations in the next message into one Observation. Set ROUND to the
latest source round and preserve source-round metadata with SOURCE_ROUND_START and
SOURCE_ROUND_END. Preserve positive decision impact and the union of all affected
targets. Preserve every source tool-call ID. Each evidence entry is one
nested EVIDENCE block with CATEGORY, TEXT, and repeated TOOL_CALL_ID fields. Use JSON
literals after '=' and do not output prose."""


class ContextMaintenanceError(RuntimeError):
    """The Runtime context contract could not be maintained."""


class _ObservationProviderError(ContextMaintenanceError):
    """The provider could not complete an asynchronous Observation request."""


class _ObservationValidationError(ContextMaintenanceError):
    """The provider response did not satisfy the Observation contract."""


def _observation_contract() -> str:
    return _OBSERVATION_CONTRACT


def _merge_observation_contract() -> str:
    return _MERGE_OBSERVATION_CONTRACT


def observation_decision_context(
    *,
    goal: str,
    execution_mode: str,
    plan_step: Mapping[str, Any] | None = None,
    completion_criterion: str | None = None,
) -> dict[str, Any]:
    """Build the loop-owned context used to assess one Tool call's impact."""
    return {
        "goal": goal,
        "execution_mode": execution_mode,
        "plan_step": dict(plan_step) if plan_step is not None else None,
        "completion_criterion": completion_criterion,
    }


@dataclass(frozen=True, slots=True)
class Evidence:
    text: str
    tool_call_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Observation:
    round: int
    confirmed_facts: tuple[Evidence, ...]
    reported_errors: tuple[Evidence, ...]
    model_inferences: tuple[Evidence, ...]
    affects_current_decision: bool
    affected_targets: tuple[str, ...]
    source_round_start: int | None = None
    source_round_end: int | None = None

    @property
    def source_tool_call_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                call_id
                for entries in (
                    self.confirmed_facts,
                    self.reported_errors,
                    self.model_inferences,
                )
                for entry in entries
                for call_id in entry.tool_call_ids
            )
        )


@dataclass(frozen=True, slots=True)
class ObservationOutcome:
    """Transient diagnostic state for one asynchronous Observation attempt."""

    round: int
    status: Literal["pending", "succeeded", "provider_error", "invalid", "cancelled"]
    error: str | None = None
    tool_call_id: str = "unknown"


@dataclass(frozen=True, slots=True)
class RawToolCall:
    tool_call_id: str
    name: str
    arguments: Mapping[str, Any]
    result: Any = None
    error: str | None = None
    lifecycle: EvidenceLifecycle = "observation"


@dataclass(frozen=True, slots=True)
class RawToolResult:
    round: int
    calls: tuple[RawToolCall, ...]


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    durable_state: Any
    observations: tuple[Observation, ...]
    raw_tool_results: tuple[RawToolResult, ...]

    def as_messages(self, *, completion_criteria_status: str | None = None) -> list[HumanMessage]:
        """Render the policy-owned layers for a model request."""
        return [HumanMessage(content=_runtime_context_prompt(self, completion_criteria_status))]


class RuntimeContextPolicy:
    """Maintain exact recent Tool rounds and compact older Observations."""

    def __init__(
        self,
        model: Any,
        *,
        budget: int = DEFAULT_CONTEXT_BUDGET,
        raw_rounds: int = RAW_ROUND_WINDOW,
        trace: Any | None = None,
    ) -> None:
        if type(budget) is not int or budget <= 0:
            raise ValueError("context budget must be a positive integer")
        if type(raw_rounds) is not int or raw_rounds <= 0:
            raise ValueError("raw_rounds must be a positive integer")
        self.model = model
        self.budget = budget
        self.raw_rounds = raw_rounds
        self._trace = trace
        self._raw: list[RawToolResult] = []
        self._observations: list[Observation] = []
        self._observation_outcomes: dict[tuple[int, str], ObservationOutcome] = {}
        self._observation_tasks: set[asyncio.Task[Any]] = set()
        self._call_order: dict[tuple[int, str], int] = {}
        self._next_call_order = 0
        self._model_uses = 0
        self._tool_rounds = 0
        self._decision_summaries: dict[tuple[int, str], Mapping[str, Any]] = {}

    @property
    def observations(self) -> tuple[Observation, ...]:
        return tuple(self._observations)

    @property
    def raw_tool_results(self) -> tuple[RawToolResult, ...]:
        return tuple(self._raw)

    @property
    def observation_outcomes(self) -> tuple[ObservationOutcome, ...]:
        """Return transient diagnostics in Observation-start order."""
        return tuple(self._observation_outcomes.values())

    def fresh_for_invocation(self) -> RuntimeContextPolicy:
        """Create an equivalent policy with no transient Tool-round history."""
        return RuntimeContextPolicy(
            self.model,
            budget=self.budget,
            raw_rounds=self.raw_rounds,
            trace=self._trace,
        )

    @property
    def model_uses(self) -> int:
        return self._model_uses

    @property
    def tool_rounds(self) -> int:
        return self._tool_rounds

    def record_model_use(self) -> None:
        """Account for an operational or final-completion model request."""
        self._model_uses += 1

    async def record_tool_round(
        self,
        round: int,
        calls: Sequence[Mapping[str, Any]],
        results: Sequence[Any],
        errors: Sequence[str | None] | None = None,
        decision_context: Mapping[str, Any] | None = None,
        suppress_observations: bool = False,
    ) -> RawToolResult:
        """Store a settled batch and observe only successful observation-class calls."""
        if len(calls) != len(results) or (errors is not None and len(calls) != len(errors)):
            raise ValueError("tool calls and settled results must have equal length")
        errors = errors or (None,) * len(results)
        raw = RawToolResult(
            round,
            tuple(
                RawToolCall(
                    str(call.get("id") or "unknown"),
                    str(call.get("name") or "unknown"),
                    dict(call.get("args") or {}),
                    result=result,
                    error=error or _result_error(result),
                    lifecycle=_native_evidence_lifecycle(
                        str(call.get("name") or "unknown"), call.get("args") or {}
                    ),
                )
                for call, result, error in zip(calls, results, errors)
            ),
        )
        self._raw.append(raw)
        self._tool_rounds += 1
        for call in raw.calls:
            self._call_order[(raw.round, call.tool_call_id)] = self._next_call_order
            self._next_call_order += 1
            if call.lifecycle == "observation":
                summary = dict(decision_context or {})
                summary["tool"] = {"name": call.name, "arguments": dict(call.arguments)}
                self._decision_summaries[(raw.round, call.tool_call_id)] = summary
        self._expire_observations()
        for call in (
            call
            for call in raw.calls
            if not suppress_observations
            and call.lifecycle == "observation"
            and call.error is None
        ):
            observation_raw = RawToolResult(raw.round, (call,))
            self._set_observation_outcome(raw.round, call.tool_call_id, "pending")
            task = asyncio.create_task(
                self._observe(
                    observation_raw,
                    self._decision_summaries.get((raw.round, call.tool_call_id), {}),
                )
            )
            self._observation_tasks.add(task)
            task.add_done_callback(self._observation_finished)
            await asyncio.sleep(0)
        return raw

    def assemble(self, durable_state: Any) -> RuntimeContext:
        """Assemble Durable State, older Observations, and recent Raw results."""
        recent_rounds = {item.round for item in self._raw[-self.raw_rounds :]}
        current_rounds = {item.round for item in self._raw[-NATIVE_LISTING_ROUND_WINDOW :]}
        observations = tuple(self._observations)
        observed_call_ids = {
            call_id
            for item in self._observations
            for call_id in item.source_tool_call_ids
        }
        negative_call_ids = {
            call_id
            for (_, call_id), outcome in self._observation_outcomes.items()
            if outcome.status == "succeeded" and call_id not in observed_call_ids
        }
        raw_results = tuple(
            RawToolResult(
                item.round,
                tuple(
                    call
                    for call in item.calls
                    if _is_visible_call(
                        call,
                        item.round,
                        current_rounds,
                        recent_rounds,
                        observed_call_ids,
                        negative_call_ids,
                    )
                ),
            )
            for item in self._raw
            if any(
                _is_visible_call(
                    call,
                    item.round,
                    current_rounds,
                    recent_rounds,
                    observed_call_ids,
                    negative_call_ids,
                )
                for call in item.calls
            )
        )
        return RuntimeContext(durable_state, observations, raw_results)

    async def maintain(self, durable_state: Any) -> RuntimeContext:
        """Assemble a window, merging oldest Observations until it fits."""
        context = self.assemble(durable_state)
        while _estimate_tokens(context) > self.budget:
            visible = [
                item
                for item in self._observations
                if item in context.observations
            ]
            if len(visible) < 2:
                raise ContextMaintenanceError("required Runtime context exceeds its budget")
            first, second = visible[:2]
            merged = await self._merge_observations(first, second)
            first_index = self._observations.index(first)
            second_index = self._observations.index(second)
            self._observations[first_index : second_index + 1] = [merged]
            context = self.assemble(durable_state)
        return context

    def _expire_observations(self) -> None:
        # Observations remain retained in history; assemble() controls whether
        # they are visible while their Raw round is in the recent window.
        return None

    async def _observe(
        self, raw: RawToolResult, decision_summary: Mapping[str, Any]
    ) -> None:
        call_id = raw.calls[0].tool_call_id
        try:
            observation = await self._request_observation(raw, decision_summary)
        except asyncio.CancelledError:
            self._set_observation_outcome(
                raw.round, call_id, "cancelled", "Observation task was cancelled"
            )
        except _ObservationProviderError as error:
            self._set_observation_outcome(raw.round, call_id, "provider_error", str(error))
        except _ObservationValidationError as error:
            self._set_observation_outcome(raw.round, call_id, "invalid", str(error))
        except ContextMaintenanceError as error:
            self._set_observation_outcome(raw.round, call_id, "invalid", str(error))
            return
        else:
            if observation.affects_current_decision:
                self._observations.append(observation)
                self._observations.sort(key=self._observation_order)
            self._set_observation_outcome(raw.round, call_id, "succeeded")

    def _observation_order(self, observation: Observation) -> tuple[int, int]:
        source_round = observation.source_round_start or observation.round
        source_order = min(
            (
                self._call_order.get((source_round, call_id), self._next_call_order)
                for call_id in observation.source_tool_call_ids
            ),
            default=self._next_call_order,
        )
        return source_order, source_round

    def _set_observation_outcome(
        self,
        round: int,
        tool_call_id: str,
        status: Literal["pending", "succeeded", "provider_error", "invalid", "cancelled"],
        error: str | None = None,
    ) -> None:
        outcome = ObservationOutcome(round, status, error, tool_call_id)
        self._observation_outcomes[(round, tool_call_id)] = outcome
        if self._trace is not None:
            callback = getattr(self._trace, "runtime_context_observation", None)
            if callback is not None:
                callback(outcome)

    def _observation_finished(self, task: asyncio.Task[Any]) -> None:
        self._observation_tasks.discard(task)

    async def _request_observation(
        self, raw: RawToolResult, decision_summary: Mapping[str, Any]
    ) -> Observation:
        payload = _raw_payload(raw)
        payload["decision_summary"] = dict(decision_summary)
        response = await self._structured_invoke(
            [
                HumanMessage(content=_observation_contract()),
                HumanMessage(content=json.dumps(payload, default=str, sort_keys=True)),
            ],
            request_kind="observation",
        )
        return _parse_observation(
            response,
            expected_round=raw.round,
            expected_tool_call_id=raw.calls[0].tool_call_id,
        )

    async def _merge_observations(self, first: Observation, second: Observation) -> Observation:
        response = await self._structured_invoke(
            [
                HumanMessage(content=_merge_observation_contract()),
                HumanMessage(
                    content=json.dumps(
                        [_observation_payload(first), _observation_payload(second)],
                        default=str,
                    )
                ),
            ],
            request_kind="observation_merge",
        )
        merged = _parse_observation(response, expected_round=second.round)
        expected_ids = set(first.source_tool_call_ids) | set(second.source_tool_call_ids)
        if not expected_ids.issubset(set(merged.source_tool_call_ids)):
            raise ContextMaintenanceError("merged Observation lost source tool-call IDs")
        return Observation(
            merged.round,
            merged.confirmed_facts,
            merged.reported_errors,
            merged.model_inferences,
            any(item.affects_current_decision for item in (first, second)),
            tuple(dict.fromkeys(first.affected_targets + second.affected_targets)),
            source_round_start=first.source_round_start or first.round,
            source_round_end=second.source_round_end or second.round,
        )

    async def _structured_invoke(
        self, messages: list[HumanMessage], *, request_kind: str
    ) -> Any:
        self._model_uses += 1
        try:
            bound = self.model.bind(tools=[])
            if self._trace is not None:
                callback = getattr(self._trace, "llm_request", None)
                runtime_context_callback = getattr(self._trace, "runtime_context_llm_request", None)
                if runtime_context_callback is not None:
                    runtime_context_callback(messages)
                elif callback is not None:
                    callback(
                        "runtime_context",
                        messages,
                        static_shape={
                            "instructions": getattr(messages[0], "content", ""),
                            "request_kind": request_kind,
                        },
                    )
                else:
                    callback = getattr(self._trace, "llm_context", None)
                    if callback is not None:
                        callback(messages)
            response = await bound.ainvoke(messages)
            if self._trace is not None:
                callback = getattr(self._trace, "llm_response", None)
                runtime_context_callback = getattr(self._trace, "runtime_context_llm_response", None)
                static_shape = {
                    "instructions": getattr(messages[0], "content", ""),
                    "request_kind": request_kind,
                }
                if runtime_context_callback is not None:
                    runtime_context_callback(response, static_shape=static_shape)
                elif callback is not None:
                    callback(response)
            return response
        except Exception as error:
            raise _ObservationProviderError(str(error) or "Runtime context model request failed") from error


def _parse_observation(
    response: Any,
    *,
    expected_round: int,
    expected_tool_call_id: str | None = None,
) -> Observation:
    content = getattr(response, "content", response)
    if not isinstance(content, str):
        raise _ObservationValidationError("Observation protocol response must be text")
    try:
        block = parse_line_protocol(content)
        if block.type != "OBSERVATION":
            raise LineProtocolError("expected BEGIN OBSERVATION")
        fields = block.fields
        round_number = _single_protocol_field(fields, "ROUND")
        if type(round_number) is not int or round_number != expected_round:
            raise ValueError("round")
        source_start = _optional_protocol_int(fields, "SOURCE_ROUND_START")
        source_end = _optional_protocol_int(fields, "SOURCE_ROUND_END")
        if (source_start is None) != (source_end is None) or (
            source_start is not None and source_start > source_end
        ):
            raise ValueError("source range")
        impact = _single_protocol_field(fields, "AFFECTS_CURRENT_DECISION")
        if type(impact) is not bool:
            raise ValueError("AFFECTS_CURRENT_DECISION")
        targets = fields.get("AFFECTED_TARGETS", [])
        valid_targets = {"confirmed_facts", "summary", "durable_state"}
        if any(target not in valid_targets for target in targets):
            raise ValueError("AFFECTED_TARGETS")
        if (impact and not targets) or (not impact and targets):
            raise ValueError("decision impact targets")
        parsed: dict[str, tuple[Evidence, ...]] = {
            "confirmed_facts": (), "reported_errors": (), "model_inferences": ()
        }
        for evidence in block.children:
            block_fields = evidence.fields
            if set(block_fields) - {"CATEGORY", "TEXT", "TOOL_CALL_ID"}:
                raise ValueError("evidence fields")
            category = _single_protocol_field(block_fields, "CATEGORY")
            if category not in parsed:
                raise ValueError("category")
            text = _single_protocol_field(block_fields, "TEXT")
            ids = block_fields.get("TOOL_CALL_ID", [])
            if not isinstance(text, str) or not text.strip() or not ids or any(not isinstance(item, str) for item in ids):
                raise ValueError("evidence")
            parsed[category] += (Evidence(text, tuple(ids)),)
        observation = Observation(
            expected_round,
            **parsed,
            affects_current_decision=impact,
            affected_targets=tuple(dict.fromkeys(targets)),
            source_round_start=source_start,
            source_round_end=source_end,
        )
        if expected_tool_call_id is not None:
            source_ids = set(observation.source_tool_call_ids)
            if source_ids and source_ids != {expected_tool_call_id}:
                raise ValueError("source tool-call attribution")
            if impact and source_ids != {expected_tool_call_id}:
                raise ValueError("positive impact requires source tool-call attribution")
            if source_start is not None and (
                source_start != expected_round or source_end != expected_round
            ):
                raise ValueError("source round attribution")
        return observation
    except (KeyError, TypeError, ValueError, LineProtocolError) as error:
        raise _ObservationValidationError(
            f"Observation Line Protocol is invalid: {error}"
        ) from error


def _single_protocol_field(fields: Mapping[str, list[Any]], name: str) -> Any:
    values = fields.get(name)
    if values is None or len(values) != 1:
        raise ValueError(name)
    return values[0]


def _optional_protocol_int(fields: Mapping[str, list[Any]], name: str) -> int | None:
    if name not in fields:
        return None
    value = _single_protocol_field(fields, name)
    if type(value) is not int:
        raise ValueError(name)
    return value


def _result_error(result: Any) -> str | None:
    if isinstance(result, Mapping) and (result.get("error") or result.get("isError") is True):
        return str(result.get("error") or result)
    return None


def _is_visible_call(
    call: RawToolCall,
    round: int,
    current_rounds: set[int],
    recent_rounds: set[int],
    observed_call_ids: set[str],
    negative_call_ids: set[str],
) -> bool:
    return (
        (call.lifecycle == "current_raw" and round in current_rounds)
        or (call.lifecycle == "recent_raw" and round in recent_rounds)
        or call.lifecycle == "permanent_raw"
        or (
            call.lifecycle == "observation"
            and call.tool_call_id not in observed_call_ids
            and (
                call.tool_call_id not in negative_call_ids
                or round in recent_rounds
            )
        )
    )


def classify_exec_command(command: Any) -> EvidenceLifecycle:
    """Classify a shell command without executing or expanding it.

    Only one top-level command from the explicitly allow-listed command sets is
    considered readable.  Any shell syntax that could make the command's
    effects uncertain receives the write-class Observation lifecycle.
    """
    if not isinstance(command, str) or not command.strip():
        return "observation"
    if _contains_uncertain_shell_syntax(command):
        return "observation"
    try:
        words = shlex.split(command, comments=False, posix=True)
    except ValueError:
        return "observation"
    if not words:
        return "observation"
    if words[0] in {"ls", "find", "fd"}:
        return "recent_raw"
    if words[0] in {"rg", "grep", "cat", "sed", "head", "tail"}:
        return "permanent_raw"
    return "observation"


def _contains_uncertain_shell_syntax(command: str) -> bool:
    quote: str | None = None
    for character in command:
        if quote is not None:
            if quote == '"' and character in {"$", "`", "\\"}:
                return True
            if character == quote:
                quote = None
            continue
        if character in {"'", '"'}:
            quote = character
        elif character in "|&;()<>`$#\\\n\r":
            return True
    return quote is not None


def _native_evidence_lifecycle(name: str, arguments: Mapping[str, Any] | None = None) -> EvidenceLifecycle:
    if name in {"list_dir", "glob"}:
        return "current_raw"
    if name in {"grep", "read_file"}:
        return "permanent_raw"
    if name in {"exec", "atom.exec"}:
        return classify_exec_command((arguments or {}).get("command"))
    return "observation"


def _evidence_payload(item: Evidence) -> dict[str, Any]:
    return {"text": item.text, "tool_call_ids": list(item.tool_call_ids)}


def _observation_payload(item: Observation) -> dict[str, Any]:
    return {
        "round": item.round,
        "affects_current_decision": item.affects_current_decision,
        "affected_targets": list(item.affected_targets),
        "source_round_start": item.source_round_start or item.round,
        "source_round_end": item.source_round_end or item.round,
        "confirmed_facts": [_evidence_payload(value) for value in item.confirmed_facts],
        "reported_errors": [_evidence_payload(value) for value in item.reported_errors],
        "model_inferences": [_evidence_payload(value) for value in item.model_inferences],
    }


def _raw_payload(item: RawToolResult) -> dict[str, Any]:
    return {"round": item.round, "calls": [{"id": call.tool_call_id, "name": call.name, "args": call.arguments, "result": call.result, "error": call.error} for call in item.calls]}


def _runtime_context_prompt(
    context: RuntimeContext, completion_criteria_status: str | None = None
) -> str:
    """Render Runtime context as explicit prompt sections instead of one JSON blob."""
    durable_state, goal, steps = _runtime_durable_state_goal_and_steps(context.durable_state)
    sections = [
        "## Runtime context",
        "",
        "### Durable state",
        _prompt_value(durable_state),
        "",
        "### Observations",
    ]
    for title, attribute in (
        ("Confirmed facts", "confirmed_facts"),
        ("Reported errors", "reported_errors"),
        ("Model inferences", "model_inferences"),
    ):
        sections.append(f"- {title}:")
        evidence = [
            entry
            for observation in context.observations
            for entry in getattr(observation, attribute)
        ]
        sections.extend(
            [f"  - {entry.text}" for entry in evidence] or ["  - (none)"]
        )

    sections.extend(("", "### Raw tool results"))
    native_groups, round_results = _group_native_read_results(context.raw_tool_results)
    for path, entries in native_groups:
        sections.append(f"#### File: {path}")
        for _, call, result in entries:
            if path == "(unfiled)":
                sections.extend(_render_raw_call(call, result))
            else:
                sections.extend(_render_code_block(result))
    for result in round_results:
        if not result.calls:
            continue
        sections.append(f"#### Round {result.round}")
        for call in result.calls:
            sections.extend(_render_raw_call(call, call.result))
    if not native_groups and not any(result.calls for result in round_results):
        sections.append("- (none)")

    sections.extend(("", "### Goal", _prompt_value(goal)))
    if steps is not None:
        sections.extend(("", "### Steps", _prompt_value(steps)))
    if completion_criteria_status is not None:
        sections.extend(("", "### Completion criteria", completion_criteria_status))
    return "\n".join(sections)


def _render_code_block(value: Any) -> list[str]:
    text = value if isinstance(value, str) else _prompt_value(value)
    fence = "```"
    while fence in text:
        fence += "`"
    return [f"{fence}text", text, fence]


def _render_raw_call(call: RawToolCall, result: Any, *, prefix: str = "") -> list[str]:
    lines = [f"- {prefix}{call.name} (tool_call_id={call.tool_call_id})"]
    lines.append(f"  - arguments: {_prompt_value(call.arguments)}")
    lines.append(f"  - result: {_prompt_value(result)}")
    if call.error is not None:
        lines.append(f"  - error: {call.error}")
    return lines


def _group_native_read_results(
    raw_results: tuple[RawToolResult, ...],
) -> tuple[list[tuple[str, list[tuple[int, RawToolCall, Any]]]], list[RawToolResult]]:
    groups: dict[str, list[tuple[int, RawToolCall, Any]]] = {}
    round_results: list[RawToolResult] = []
    for raw in raw_results:
        other_calls: list[RawToolCall] = []
        for call in raw.calls:
            if call.name not in {"grep", "read_file"} or call.lifecycle != "permanent_raw":
                other_calls.append(call)
                continue
            entries = _native_read_entries(call)
            if call.error is not None or entries is None:
                groups.setdefault("(unfiled)", []).append((raw.round, call, call.result))
                continue
            for path, result in entries:
                groups.setdefault(path, []).append((raw.round, call, result))
        round_results.append(RawToolResult(raw.round, tuple(other_calls)))
    ordered = []
    for path, entries in groups.items():
        entries.sort(key=lambda entry: entry[0])
        ordered.append((path, entries))
    return ordered, round_results


def _native_read_entries(call: RawToolCall) -> list[tuple[str, Any]] | None:
    value = call.arguments.get("path")
    if isinstance(value, str) and value.strip():
        return [(value.strip(), call.result)]
    if call.name == "grep" and isinstance(call.result, str):
        lines = [line for line in call.result.splitlines() if line]
        parsed = [
            (line.split(":", 1)[0].strip(), line.split(":", 1)[1])
            for line in lines
            if ":" in line
        ]
        paths = [path for path, _ in parsed]
        if lines and len(paths) == len(lines) and all(paths):
            by_path: dict[str, list[str]] = {}
            for path, content in parsed:
                by_path.setdefault(path, []).append(content)
            return [(path, "\n".join(contents)) for path, contents in by_path.items()]
    return None


def _runtime_durable_state_goal_and_steps(value: Any) -> tuple[Any, Any, Any | None]:
    """Move a Plan execution's steps to the final Runtime context section."""
    if not isinstance(value, Mapping):
        return value, None, None
    if "goal" in value:
        durable_state = dict(value)
        goal = durable_state.pop("goal")
        return durable_state, goal, None
    agent_state = value.get("agent_state")
    if isinstance(agent_state, Mapping) and "goal" in agent_state:
        durable_state = dict(value)
        displayed_agent_state = dict(agent_state)
        goal = displayed_agent_state.pop("goal")
        steps = _extract_plan_steps(displayed_agent_state)
        durable_state["agent_state"] = displayed_agent_state
        return durable_state, goal, steps
    return value, None, None


def _extract_plan_steps(agent_state: dict[str, Any]) -> list[dict[str, Any]] | None:
    """Remove plan steps while retaining their revision association."""
    plan_history = agent_state.get("plan_history")
    if not isinstance(plan_history, list):
        return None

    displayed_history: list[Any] = []
    steps: list[dict[str, Any]] = []
    for plan in plan_history:
        if not isinstance(plan, Mapping) or "steps" not in plan:
            displayed_history.append(plan)
            continue
        displayed_plan = dict(plan)
        plan_steps = displayed_plan.pop("steps")
        displayed_history.append(displayed_plan)
        steps.append({"revision": plan.get("revision"), "steps": plan_steps})
    agent_state["plan_history"] = displayed_history
    return steps


def _evidence_section(title: str, values: tuple[Evidence, ...]) -> list[str]:
    if not values:
        return [f"- {title}: (none)"]
    lines = [f"- {title}:"]
    for value in values:
        call_ids = ", ".join(value.tool_call_ids) or "none"
        lines.append(f"  - {value.text} (tool_call_ids: {call_ids})")
    return lines


def _observation_source(observation: Observation) -> str:
    if observation.source_round_start is None and observation.source_round_end is None:
        return ""
    start = observation.source_round_start or observation.round
    end = observation.source_round_end or observation.round
    return f" (source rounds {start}-{end})"


def _observation_start_round(observation: Observation) -> int:
    return observation.source_round_start or observation.round


def _prompt_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _estimate_tokens(value: Any) -> int:
    return max(1, len(json.dumps(value, default=str, ensure_ascii=False)) // 4)
