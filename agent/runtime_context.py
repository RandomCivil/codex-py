"""Layered Runtime context for tool-capable model loops.

The policy deliberately keeps transient tool evidence separate from durable
state.  Callers may use :meth:`assemble` for the current in-memory window and
``maintain`` when budget-driven Observation merging is needed.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from langchain_core.messages import HumanMessage

from llm.line_protocol import LineProtocolError, parse_line_protocol


DEFAULT_CONTEXT_BUDGET = 128_000
RAW_ROUND_WINDOW = 3


_OBSERVATION_CONTRACT = """Return exactly one UTF-8 Line Protocol block:
BEGIN OBSERVATION ... END OBSERVATION.
Create the Observation for the settled Tool round in the next message.
Classify tool-confirmed output as confirmed_facts, every raw error as reported_errors,
and interpretations as model_inferences. Set ROUND to the source round. Every evidence
entry is one nested EVIDENCE block with CATEGORY, TEXT, and one or more repeated
TOOL_CALL_ID fields. Use JSON literals after '=' and do not output prose."""

_MERGE_OBSERVATION_CONTRACT = """Return exactly one UTF-8 Line Protocol block:
BEGIN OBSERVATION ... END OBSERVATION.
Merge the Observations in the next message into one Observation. Set ROUND to the
latest source round and preserve source-round metadata with SOURCE_ROUND_START and
SOURCE_ROUND_END. Preserve every source tool-call ID. Each evidence entry is one
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


@dataclass(frozen=True, slots=True)
class RawToolCall:
    tool_call_id: str
    name: str
    arguments: Mapping[str, Any]
    result: Any = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class RawToolResult:
    round: int
    calls: tuple[RawToolCall, ...]


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    durable_state: Any
    observations: tuple[Observation, ...]
    raw_tool_results: tuple[RawToolResult, ...]

    def as_messages(self) -> list[HumanMessage]:
        """Render the policy-owned layers for a model request."""
        return [HumanMessage(content=_runtime_context_prompt(self))]


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
        self._observation_outcomes: dict[int, ObservationOutcome] = {}
        self._observation_tasks: set[asyncio.Task[Any]] = set()
        self._model_uses = 0
        self._tool_rounds = 0

    @property
    def observations(self) -> tuple[Observation, ...]:
        return tuple(self._observations)

    @property
    def raw_tool_results(self) -> tuple[RawToolResult, ...]:
        return tuple(self._raw)

    @property
    def observation_outcomes(self) -> tuple[ObservationOutcome, ...]:
        """Return transient diagnostics in settled Tool-round order."""
        return tuple(self._observation_outcomes[item.round] for item in self._raw)

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
    ) -> RawToolResult:
        """Store a settled batch and start its best-effort Observation."""
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
                )
                for call, result, error in zip(calls, results, errors)
            ),
        )
        self._raw.append(raw)
        self._tool_rounds += 1
        self._expire_observations()
        self._set_observation_outcome(raw.round, "pending")
        task = asyncio.create_task(self._observe(raw))
        self._observation_tasks.add(task)
        task.add_done_callback(self._observation_finished)
        await asyncio.sleep(0)
        return raw

    def assemble(self, durable_state: Any) -> RuntimeContext:
        """Assemble Durable State, older Observations, and recent Raw results."""
        recent_rounds = {item.round for item in self._raw[-self.raw_rounds :]}
        observations = tuple(
            item
            for item in self._observations
            if not all(
                round_number in recent_rounds
                for round_number in range(
                    item.source_round_start or item.round,
                    (item.source_round_end or item.round) + 1,
                )
            )
        )
        observed_rounds = {
            round_number
            for item in self._observations
            for round_number in range(
                item.source_round_start or item.round,
                (item.source_round_end or item.round) + 1,
            )
        }
        raw_results = tuple(
            item
            for item in self._raw
            if item.round in recent_rounds or item.round not in observed_rounds
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

    async def _observe(self, raw: RawToolResult) -> None:
        try:
            observation = await self._request_observation(raw)
        except asyncio.CancelledError:
            self._set_observation_outcome(raw.round, "cancelled", "Observation task was cancelled")
        except _ObservationProviderError as error:
            self._set_observation_outcome(raw.round, "provider_error", str(error))
        except _ObservationValidationError as error:
            self._set_observation_outcome(raw.round, "invalid", str(error))
        except ContextMaintenanceError as error:
            self._set_observation_outcome(raw.round, "invalid", str(error))
            return
        else:
            self._observations.append(observation)
            self._observations.sort(key=_observation_start_round)
            self._set_observation_outcome(raw.round, "succeeded")

    def _set_observation_outcome(
        self,
        round: int,
        status: Literal["pending", "succeeded", "provider_error", "invalid", "cancelled"],
        error: str | None = None,
    ) -> None:
        self._observation_outcomes[round] = ObservationOutcome(round, status, error)
        if self._trace is not None:
            callback = getattr(self._trace, "runtime_context_observation", None)
            if callback is not None:
                callback(self._observation_outcomes[round])

    def _observation_finished(self, task: asyncio.Task[Any]) -> None:
        self._observation_tasks.discard(task)

    async def _request_observation(self, raw: RawToolResult) -> Observation:
        response = await self._structured_invoke(
            [
                HumanMessage(content=_observation_contract()),
                HumanMessage(content=json.dumps(_raw_payload(raw), default=str)),
            ],
            request_kind="observation",
        )
        return _parse_observation(response, expected_round=raw.round)

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


def _parse_observation(response: Any, *, expected_round: int) -> Observation:
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
        return Observation(expected_round, **parsed, source_round_start=source_start, source_round_end=source_end)
    except (KeyError, TypeError, ValueError, LineProtocolError) as error:
        raise _ObservationValidationError("Observation Line Protocol is invalid") from error


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


def _evidence_payload(item: Evidence) -> dict[str, Any]:
    return {"text": item.text, "tool_call_ids": list(item.tool_call_ids)}


def _observation_payload(item: Observation) -> dict[str, Any]:
    return {
        "round": item.round,
        "source_round_start": item.source_round_start or item.round,
        "source_round_end": item.source_round_end or item.round,
        "confirmed_facts": [_evidence_payload(value) for value in item.confirmed_facts],
        "reported_errors": [_evidence_payload(value) for value in item.reported_errors],
        "model_inferences": [_evidence_payload(value) for value in item.model_inferences],
    }


def _raw_payload(item: RawToolResult) -> dict[str, Any]:
    return {"round": item.round, "calls": [{"id": call.tool_call_id, "name": call.name, "args": call.arguments, "result": call.result, "error": call.error} for call in item.calls]}


def _runtime_context_prompt(context: RuntimeContext) -> str:
    """Render Runtime context as explicit prompt sections instead of one JSON blob."""
    sections = [
        "## Runtime context",
        "",
        "### Durable state",
        _prompt_value(context.durable_state),
        "",
        "### Observations",
    ]
    if context.observations:
        for observation in context.observations:
            source = _observation_source(observation)
            sections.extend((f"#### Round {observation.round}{source}",))
            sections.extend(_evidence_section("Confirmed facts", observation.confirmed_facts))
            sections.extend(_evidence_section("Reported errors", observation.reported_errors))
            sections.extend(_evidence_section("Model inferences", observation.model_inferences))
    else:
        sections.append("- (none)")

    sections.extend(("", "### Raw tool results"))
    if context.raw_tool_results:
        for result in context.raw_tool_results:
            sections.append(f"#### Round {result.round}")
            if not result.calls:
                sections.append("- (no tool calls)")
            for call in result.calls:
                sections.append(
                    f"- {call.name} (tool_call_id={call.tool_call_id})"
                )
                sections.append(f"  - arguments: {_prompt_value(call.arguments)}")
                sections.append(f"  - result: {_prompt_value(call.result)}")
                if call.error is not None:
                    sections.append(f"  - error: {call.error}")
    else:
        sections.append("- (none)")
    return "\n".join(sections)


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
