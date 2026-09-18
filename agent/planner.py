import json
from collections.abc import AsyncIterator
from typing import Any, Mapping, Protocol

from llm.response_format import ResponseFormat, require_response_format
from memory.state import AgentState, Plan, PlanStep


class TextStream(Protocol):
    def stream_text(
        self,
        input: str,
        *,
        instructions: str | None = None,
        tools: Any = None,
        text_format: Mapping[str, Any],
    ) -> AsyncIterator[str]: ...


class PlanningValidationError(ValueError):
    """The model did not return a valid structured Plan."""


class Planner:
    _TEXT_FORMAT = {
        "type": "json_schema",
        "name": "agent_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["revision", "goal", "steps"],
            "properties": {
                "revision": {"type": "integer"},
                "goal": {"type": "string", "minLength": 1},
                "steps": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["id", "intent", "completion_criterion"],
                        "properties": {
                            "id": {"type": "string", "minLength": 1, "pattern": "\\S"},
                            "intent": {"type": "string", "minLength": 1, "pattern": "\\S"},
                            "completion_criterion": {"type": "string", "minLength": 1, "pattern": "\\S"},
                        },
                    },
                },
            },
        },
    }
    _INSTRUCTIONS = (
        "Return only strict JSON for the complete Plan. Do not use markdown or prose. "
        "The JSON must contain exactly revision, goal, and steps. Each step must "
        "contain exactly id, intent, and completion_criterion."
    )

    def __init__(self, text_stream: TextStream, trace: Any | None = None, *, response_format: ResponseFormat = "json_schema") -> None:
        self._text_stream = text_stream
        self._trace = trace
        self._response_format = require_response_format(response_format)

    async def plan(self, state: AgentState) -> Plan:
        expected_revision = len(state.plan_history) + 1
        if self._trace is not None:
            self._trace.plan("started", expected_revision)
        if expected_revision > 3:
            raise PlanningValidationError(
                "planning cannot exceed the three-revision goal budget"
            )
        request = json.dumps(_state_payload(state), ensure_ascii=False, sort_keys=True)
        text_format = self._TEXT_FORMAT if self._response_format == "json_schema" else {"type": "json_object"}
        output = "".join(
            [
                chunk
                async for chunk in self._text_stream.stream_text(
                    request,
                    instructions=self._INSTRUCTIONS,
                    tools=None,
                    text_format=text_format,
                )
            ]
        )
        plan = _parse_plan(output, state.goal, expected_revision)
        if self._trace is not None:
            self._trace.plan("completed", plan.revision)
        return plan


def _state_payload(state: AgentState) -> dict[str, Any]:
    return {
        "goal": state.goal,
        "memory_summary": state.memory_summary,
        "plan_history": [
            {
                "revision": plan.revision,
                "goal": plan.goal,
                "steps": [
                    {
                        "id": step.id,
                        "intent": step.intent,
                        "completion_criterion": step.completion_criterion,
                    }
                    for step in plan.steps
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


def _parse_plan(output: str, goal: str, expected_revision: int) -> Plan:
    try:
        document = json.loads(output)
    except json.JSONDecodeError as error:
        raise PlanningValidationError("planning output must be strict JSON") from error

    if not isinstance(document, dict):
        raise PlanningValidationError("planning output must be a JSON object")
    if set(document) != {"revision", "goal", "steps"}:
        raise PlanningValidationError("plan must contain exactly revision, goal, and steps")
    revision = document["revision"]
    if type(revision) is float and revision.is_integer():
        revision = int(revision)
    elif isinstance(revision, str):
        try:
            parsed_revision = float(revision.strip())
        except ValueError:
            parsed_revision = None
        if parsed_revision is not None and parsed_revision.is_integer():
            revision = int(parsed_revision)
    if type(revision) is not int or revision != expected_revision:
        raise PlanningValidationError(
            f"plan revision must be the next sequential revision ({expected_revision})"
        )
    if document["goal"] != goal:
        raise PlanningValidationError("plan goal must match agent state goal")
    if not isinstance(document["steps"], list):
        raise PlanningValidationError("plan steps must be a JSON array")

    steps = []
    for raw_step in document["steps"]:
        if not isinstance(raw_step, dict) or set(raw_step) != {
            "id",
            "intent",
            "completion_criterion",
        }:
            raise PlanningValidationError(
                "each plan step must contain exactly id, intent, and completion_criterion"
            )
        try:
            step_id = raw_step["id"]
            # Some OpenAI-compatible providers ignore the string type in the
            # schema and emit ordinal step IDs as JSON numbers. IDs are opaque
            # strings in the domain model, so normalize that harmless variant
            # at the Planner boundary before validation.
            if type(step_id) is int:
                step_id = str(step_id)
            steps.append(
                PlanStep(
                    step_id,
                    raw_step["intent"],
                    raw_step["completion_criterion"],
                )
            )
        except (TypeError, ValueError) as error:
            raise PlanningValidationError("plan contains an invalid step") from error

    try:
        return Plan(expected_revision, goal, tuple(steps))
    except (TypeError, ValueError) as error:
        raise PlanningValidationError("plan violates its invariants") from error
