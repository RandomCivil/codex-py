import json
from typing import Any, Mapping

from llm.line_protocol import PLAN_INSTRUCTIONS, LineProtocolError, decode_plan
from llm.text_stream import TextModel
from memory.state import AgentState, Plan, PlanStep


class PlanningValidationError(ValueError):
    """The model did not return a valid structured Plan."""


class Planner:
    _PLAN_INSTRUCTIONS = PLAN_INSTRUCTIONS

    def __init__(self, text_completion: TextModel, trace: Any | None = None, *, stream: bool = False) -> None:
        self._text_completion = text_completion
        self._trace = trace
        self._stream = stream

    async def plan(self, state: AgentState, *, conversation_input: Any = None) -> Plan:
        expected_revision = len(state.plan_history) + 1
        if self._trace is not None:
            self._trace.plan("started", expected_revision)
        if expected_revision > 3:
            raise PlanningValidationError(
                "planning cannot exceed the three-revision goal budget"
            )
        request = json.dumps(
            _state_payload(state, conversation_input=conversation_input), ensure_ascii=False, sort_keys=True
        )
        if self._stream:
            output = "".join(
                [
                    chunk
                    async for chunk in self._text_completion.stream_text(
                        request,
                        instructions=self._PLAN_INSTRUCTIONS,
                        tools=None,
                    )
                ]
            )
        else:
            output = await self._text_completion.complete_text(
                request,
                instructions=self._PLAN_INSTRUCTIONS,
                tools=None,
            )
        try:
            plan = decode_plan(output, goal=state.goal, expected_revision=expected_revision)
        except LineProtocolError as error:
            raise PlanningValidationError(str(error)) from error
        if self._trace is not None:
            self._trace.plan("completed", plan.revision)
        return plan


def _state_payload(state: AgentState, *, conversation_input: Any = None) -> dict[str, Any]:
    payload = {
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
    if conversation_input is not None:
        history = (
            conversation_input["history"]
            if isinstance(conversation_input, Mapping)
            else conversation_input.history
        )
        current_input = (
            conversation_input["current_input"]
            if isinstance(conversation_input, Mapping)
            else conversation_input.current_input
        )
        payload["conversation_input"] = {
            "history": list(history),
            "current_input": current_input,
        }
    return payload


def _parse_plan(output: str, goal: str, expected_revision: int) -> Plan:
    try:
        return decode_plan(output, goal=goal, expected_revision=expected_revision)
    except LineProtocolError as error:
        raise PlanningValidationError(str(error)) from error
