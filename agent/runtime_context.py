"""Layered Runtime context for tool-capable model loops.

The policy deliberately keeps transient tool evidence separate from durable
state.  Callers may use :meth:`assemble` for the current in-memory window and
``maintain`` when budget-driven Observation merging is needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from langchain_core.messages import HumanMessage

from llm.response_format import ResponseFormat, chat_response_format


DEFAULT_CONTEXT_BUDGET = 128_000
RAW_ROUND_WINDOW = 3


_OBSERVATION_JSON_FEW_SHOT = """Return exactly one JSON object and no Markdown.

Example input tool round:
{"round": 7, "calls": [{"id": "call-7", "name": "read_file", "args": {"path": "VERSION"}, "result": "1.4.0", "error": null}]}

Example JSON output:
{"round": 7, "confirmed_facts": [{"text": "VERSION is 1.4.0", "tool_call_ids": ["call-7"]}], "reported_errors": [], "model_inferences": []}

Keep the output fields exactly as shown by the configured schema."""


class ContextMaintenanceError(RuntimeError):
    """The Runtime context contract could not be maintained."""


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
        response_format: ResponseFormat = "json_schema",
        trace: Any | None = None,
    ) -> None:
        if type(budget) is not int or budget <= 0:
            raise ValueError("context budget must be a positive integer")
        if type(raw_rounds) is not int or raw_rounds <= 0:
            raise ValueError("raw_rounds must be a positive integer")
        if response_format not in {"json_schema", "json_object"}:
            raise ValueError("response_format must be json_schema or json_object")
        self.model = model
        self.budget = budget
        self.raw_rounds = raw_rounds
        self._response_format = response_format
        self._trace = trace
        self._raw: list[RawToolResult] = []
        self._observations: list[Observation] = []
        self._model_uses = 0
        self._tool_rounds = 0

    @property
    def observations(self) -> tuple[Observation, ...]:
        return tuple(self._observations)

    @property
    def raw_tool_results(self) -> tuple[RawToolResult, ...]:
        return tuple(self._raw)

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
    ) -> Observation:
        """Store a settled batch and generate its strict Observation."""
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
        observation = await self._request_observation(raw)
        self._raw.append(raw)
        self._observations.append(observation)
        self._tool_rounds += 1
        self._expire_observations()
        return observation

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
        return RuntimeContext(durable_state, observations, tuple(self._raw[-self.raw_rounds :]))

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

    async def _request_observation(self, raw: RawToolResult) -> Observation:
        response = await self._structured_invoke(
            [
                HumanMessage(
                    content=json.dumps(
                        {
                            "instruction": (
                                "Create the Observation for this settled Tool round. "
                                "Classify tool-confirmed output as confirmed_facts, every raw "
                                "error as reported_errors, and interpretations as model_inferences. "
                                "Every entry must retain the relevant tool_call_ids.\n\n"
                                + _OBSERVATION_JSON_FEW_SHOT
                            ),
                            "tool_round": _raw_payload(raw),
                        },
                        default=str,
                    )
                )
            ]
        )
        return _parse_observation(response, expected_round=raw.round)

    async def _merge_observations(self, first: Observation, second: Observation) -> Observation:
        response = await self._structured_invoke(
            [
                HumanMessage(
                    content=json.dumps(
                        {
                            "instruction": (
                                "Merge these Observations while retaining every source tool-call ID.\n\n"
                                + _OBSERVATION_JSON_FEW_SHOT
                            ),
                            "merge_observations": [_observation_payload(first), _observation_payload(second)],
                        },
                        default=str,
                    )
                )
            ]
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

    async def _structured_invoke(self, messages: list[HumanMessage]) -> Any:
        self._model_uses += 1
        schema = _observation_response_format(self._response_format)
        try:
            bound = self.model.bind(tools=[], response_format=schema)
            if self._trace is not None:
                callback = getattr(self._trace, "llm_context", None)
                if callback is not None:
                    callback(messages)
            response = await bound.ainvoke(messages)
            if self._trace is not None:
                callback = getattr(self._trace, "llm_response", None)
                if callback is not None:
                    callback(response)
            return response
        except Exception as error:
            raise ContextMaintenanceError("Runtime context model request failed") from error


def _observation_response_format(response_format: ResponseFormat) -> dict[str, Any]:
    """Use the owning component's configured Structured-output mode."""
    return chat_response_format(
        response_format,
        name="tool_round_observation",
        schema={
            "type": "object",
            "additionalProperties": False,
            "required": ["round", "confirmed_facts", "reported_errors", "model_inferences"],
            "properties": {
                "round": {"type": "integer"},
                **{key: {"type": "array", "items": {"$ref": "#/$defs/evidence"}} for key in ("confirmed_facts", "reported_errors", "model_inferences")},
            },
            "$defs": {
                "evidence": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text", "tool_call_ids"],
                    "properties": {"text": {"type": "string", "minLength": 1}, "tool_call_ids": {"type": "array", "items": {"type": "string"}}},
                }
            },
        },
    )


def _parse_observation(response: Any, *, expected_round: int) -> Observation:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError as error:
            raise ContextMaintenanceError("Observation was not valid JSON") from error
    if not isinstance(content, Mapping):
        raise ContextMaintenanceError("Observation was not an object")
    required = {"round", "confirmed_facts", "reported_errors", "model_inferences"}
    if (
        set(content) != required
        or type(content["round"]) is not int
        or content["round"] != expected_round
    ):
        raise ContextMaintenanceError("Observation schema or round is invalid")
    try:
        parsed = {}
        for key in ("confirmed_facts", "reported_errors", "model_inferences"):
            if not isinstance(content[key], list):
                raise TypeError
            entries = []
            for item in content[key]:
                if (
                    not isinstance(item, Mapping)
                    or set(item) != {"text", "tool_call_ids"}
                    or not isinstance(item["text"], str)
                    or not isinstance(item["tool_call_ids"], list)
                    or any(not isinstance(call_id, str) for call_id in item["tool_call_ids"])
                ):
                    raise TypeError
                entries.append(Evidence(item["text"], tuple(item["tool_call_ids"])))
            parsed[key] = tuple(entries)
    except (KeyError, TypeError):
        raise ContextMaintenanceError("Observation evidence is invalid") from None
    if any(not evidence.text.strip() for values in parsed.values() for evidence in values):
        raise ContextMaintenanceError("Observation evidence is invalid")
    return Observation(expected_round, **parsed)


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


def _prompt_value(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _estimate_tokens(value: Any) -> int:
    return max(1, len(json.dumps(value, default=str, ensure_ascii=False)) // 4)
