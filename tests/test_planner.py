import asyncio
import json

import pytest

from agent import Planner, PlanningValidationError
from memory.state import AgentState, Plan, PlanStep, StepExecution


class TextStream:
    def __init__(self, *chunks: str) -> None:
        self.chunks = chunks
        self.request = None

    async def stream_text(self, input, *, instructions=None, tools=None, text_format=None):
        self.request = {
            "input": input,
            "instructions": instructions,
            "tools": tools,
            "text_format": text_format,
        }
        for chunk in self.chunks:
            yield chunk


class RetryingTextStream:
    def __init__(self, *outputs: str) -> None:
        self.outputs = iter(outputs)
        self.requests = []

    async def stream_text(self, input, *, instructions=None, tools=None, text_format=None):
        self.requests.append(
            {
                "input": input,
                "instructions": instructions,
                "tools": tools,
                "text_format": text_format,
            }
        )
        yield next(self.outputs)


def test_planner_builds_initial_plan_from_strict_json_text_stream():
    text_stream = TextStream(
        '{"revision":1,"goal":"Prepare release",'
        '"steps":[{"id":"inspect","intent":"Inspect the release",'
        '"completion_criterion":"Release risks are listed"},'
        '{"id":"publish","intent":"Publish the release",'
        '"completion_criterion":"The release is available"}]}'
    )
    state = AgentState("Prepare release", memory_summary="Version 2.0")

    plan = asyncio.run(Planner(text_stream).plan(state))

    assert plan.revision == 1
    assert plan.goal == "Prepare release"
    assert [step.id for step in plan.steps] == ["inspect", "publish"]
    assert plan.steps[1].completion_criterion == "The release is available"


def test_planner_normalizes_numeric_step_ids_from_compatible_models():
    text_stream = TextStream(
        '{"revision":1,"goal":"Prepare release",'
        '"steps":[{"id":1,"intent":"Inspect the release",'
        '"completion_criterion":"Release risks are listed"}]}'
    )
    state = AgentState("Prepare release")

    plan = asyncio.run(Planner(text_stream).plan(state))

    assert plan.steps[0].id == "1"


def test_planner_uses_agent_state_goal_when_model_paraphrases_it():
    text_stream = TextStream(
        '{"revision":1,"goal":"梳理项目中 React 组件的层级关系",'
        '"steps":[{"id":"inspect","intent":"Find components",'
        '"completion_criterion":"Components are listed"}]}'
    )

    plan = asyncio.run(Planner(text_stream).plan(AgentState("项目里react组件的层级关系")))

    assert plan.goal == "项目里react组件的层级关系"


@pytest.mark.parametrize("revision", (1.0, "1.0"))
def test_planner_normalizes_integral_revision_from_compatible_models(revision):
    text_stream = TextStream(
        '{"revision":' + json.dumps(revision) + ','
        '"goal":"Prepare release","steps":[{"id":"1","intent":"Inspect the release",'
        '"completion_criterion":"Release risks are listed"}]}'
    )

    plan = asyncio.run(Planner(text_stream).plan(AgentState("Prepare release")))

    assert plan.revision == 1


def test_planner_sends_agent_state_to_text_stream_without_tools():
    text_stream = TextStream('{"revision":1,"goal":"Prepare release","steps":[{"id":"one","intent":"Do one","completion_criterion":"One is done"}]}')
    state = AgentState("Prepare release", memory_summary="Version 2.0")

    asyncio.run(Planner(text_stream).plan(state))

    assert '"goal": "Prepare release"' in text_stream.request["input"]
    assert '"memory_summary": "Version 2.0"' in text_stream.request["input"]
    assert text_stream.request["tools"] is None
    assert text_stream.request["text_format"] == Planner._TEXT_FORMAT


def test_planner_keeps_conversation_history_separate_from_the_agent_goal():
    text_stream = TextStream(
        '{"revision":1,"goal":"Continue","steps":[{"id":"one","intent":"Continue","completion_criterion":"Done"}]}'
    )
    conversation_input = {"history": [{"sequence": 1, "answer": "Earlier"}], "current_input": "Continue"}

    asyncio.run(Planner(text_stream).plan(AgentState("Continue"), conversation_input=conversation_input))

    request = json.loads(text_stream.request["input"])
    assert request["goal"] == "Continue"
    assert request["conversation_input"] == conversation_input


def test_planner_passes_selected_response_format_to_text_stream():
    text_stream = TextStream('{"revision":1,"goal":"Prepare release","steps":[{"id":"one","intent":"Do one","completion_criterion":"One is done"}]}')

    asyncio.run(Planner(text_stream, response_format="json_object").plan(AgentState("Prepare release")))

    assert text_stream.request["text_format"]["type"] == "json_object"


def test_planner_does_not_retry_when_a_provider_ignores_the_json_schema_contract():
    text_stream = RetryingTextStream(
        "1. Inspect the repository.\n2. Update the README.",
        '{"revision":1,"goal":"Prepare release","steps":[{"id":"one","intent":"Do one","completion_criterion":"One is done"}]}',
    )

    with pytest.raises(PlanningValidationError, match="strict JSON"):
        asyncio.run(Planner(text_stream).plan(AgentState("Prepare release")))

    assert len(text_stream.requests) == 1


@pytest.mark.parametrize(
    "response_format",
    [
        "json_schema",
        "json_object",
    ],
)
def test_planner_repeats_the_strict_plan_contract_in_static_instructions(response_format):
    text_stream = TextStream(
        '{"revision":1,"goal":"Prepare release","steps":['
        '{"id":"one","intent":"Do one","completion_criterion":"One is done"}]}'
    )

    asyncio.run(
        Planner(text_stream, response_format=response_format).plan(AgentState("Prepare release"))
    )

    instructions = text_stream.request["instructions"]
    assert "Return only strict JSON" in instructions
    assert "exactly revision, goal, and steps" in instructions
    assert "exactly id, intent, and completion_criterion" in instructions
    assert "Prepare release" not in instructions


def test_planner_builds_next_revision_without_mutating_prior_plan_or_executions():
    first = Plan(1, "Prepare release", (PlanStep("inspect", "Inspect", "Risks listed"),))
    state = AgentState(
        "Prepare release",
        plan_history=(first,),
        step_executions=(StepExecution(1, "inspect", "failed", error="Checks failed"),),
    )
    text_stream = TextStream(
        '{"revision":2,"goal":"Prepare release","steps":['
        '{"id":"repair","intent":"Repair the release",'
        '"completion_criterion":"Checks pass"}]}'
    )

    replanned = asyncio.run(Planner(text_stream).plan(state))

    assert replanned.revision == 2
    assert replanned.goal == "Prepare release"
    assert [step.id for step in replanned.steps] == ["repair"]
    assert state.plan_history == (first,)
    assert state.step_executions[0].status == "failed"


def test_planner_rejects_non_json_model_output():
    text_stream = TextStream("Here is your plan: inspect, then publish.")

    with pytest.raises(PlanningValidationError, match="strict JSON"):
        asyncio.run(Planner(text_stream).plan(AgentState("Prepare release")))


def test_planner_rejects_a_plan_with_unknown_fields():
    text_stream = TextStream(
        '{"revision":1,"goal":"Prepare release","steps":['
        '{"id":"one","intent":"Do one","completion_criterion":"One is done",'
        '"tool":"shell"}]}'
    )

    with pytest.raises(PlanningValidationError, match="exactly"):
        asyncio.run(Planner(text_stream).plan(AgentState("Prepare release")))


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ('{"revision":1,"goal":"Prepare release"}', "exactly"),
        ('{"revision":1,"goal":"Prepare release","steps":[],"extra":true}', "exactly"),
    ],
)
def test_planner_rejects_missing_or_unknown_top_level_fields(output, message):
    with pytest.raises(PlanningValidationError, match=message):
        asyncio.run(Planner(TextStream(output)).plan(AgentState("Prepare release")))


def test_planner_rejects_a_revision_that_is_not_next_in_sequence():
    first = Plan(1, "Prepare release", (PlanStep("one", "Do one", "One is done"),))
    output = '{"revision":3,"goal":"Prepare release","steps":[{"id":"two","intent":"Do two","completion_criterion":"Two is done"}]}'

    with pytest.raises(PlanningValidationError, match="next sequential"):
        asyncio.run(Planner(TextStream(output)).plan(AgentState("Prepare release", (first,))))


@pytest.mark.parametrize("revision", ("true", "1.5"))
def test_planner_rejects_a_revision_with_a_non_integer_json_type(revision):
    output = (
        '{"revision":' + revision + ',"goal":"Prepare release","steps":['
        '{"id":"one","intent":"Do one","completion_criterion":"One is done"}]}'
    )

    with pytest.raises(PlanningValidationError, match="next sequential"):
        asyncio.run(Planner(TextStream(output)).plan(AgentState("Prepare release")))


def test_planner_rejects_an_empty_replanned_plan():
    first = Plan(1, "Prepare release", (PlanStep("one", "Do one", "One is done"),))

    with pytest.raises(PlanningValidationError, match="invariants"):
        asyncio.run(
            Planner(TextStream('{"revision":2,"goal":"Prepare release","steps":[]}'))
            .plan(AgentState("Prepare release", (first,)))
        )


def test_planner_rejects_duplicate_replanned_step_ids():
    first = Plan(1, "Prepare release", (PlanStep("one", "Do one", "One is done"),))
    output = (
        '{"revision":2,"goal":"Prepare release","steps":['
        '{"id":"same","intent":"Do one","completion_criterion":"One is done"},'
        '{"id":"same","intent":"Do another","completion_criterion":"Another is done"}]}'
    )

    with pytest.raises(PlanningValidationError, match="invariants"):
        asyncio.run(Planner(TextStream(output)).plan(AgentState("Prepare release", (first,))))


def test_planner_rejects_a_fourth_revision_before_requesting_model_output():
    plans = tuple(
        Plan(revision, "Prepare release", (PlanStep(str(revision), "Do it", "It is done"),))
        for revision in (1, 2, 3)
    )
    text_stream = TextStream("not JSON")

    with pytest.raises(PlanningValidationError, match="three-revision"):
        asyncio.run(Planner(text_stream).plan(AgentState("Prepare release", plans)))

    assert text_stream.request is None
