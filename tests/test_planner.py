import asyncio

import pytest

from agent import Planner, PlanningValidationError
from memory.state import AgentState, Plan, PlanStep, StepExecution


class TextStream:
    def __init__(self, *chunks: str) -> None:
        self.chunks = chunks
        self.request = None

    async def stream_text(self, input, *, instructions=None, tools=None):
        self.request = {
            "input": input,
            "instructions": instructions,
            "tools": tools,
        }
        for chunk in self.chunks:
            yield chunk


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


def test_planner_sends_agent_state_to_text_stream_without_tools():
    text_stream = TextStream('{"revision":1,"goal":"Prepare release","steps":[{"id":"one","intent":"Do one","completion_criterion":"One is done"}]}')
    state = AgentState("Prepare release", memory_summary="Version 2.0")

    asyncio.run(Planner(text_stream).plan(state))

    assert '"goal": "Prepare release"' in text_stream.request["input"]
    assert '"memory_summary": "Version 2.0"' in text_stream.request["input"]
    assert text_stream.request["tools"] is None


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


@pytest.mark.parametrize("revision", ("true", "1.0"))
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
