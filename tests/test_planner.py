import asyncio
import json

import pytest

from agent import Planner, PlanningValidationError
from memory.state import AgentState, Plan, PlanStep


def plan_document(*, revision=1, goal="Prepare release", step_id="inspect"):
    return (
        "BEGIN PLAN\n"
        f"REVISION={revision}\nGOAL={json.dumps(goal)}\n"
        "BEGIN STEP\n"
        f"ID={json.dumps(step_id)}\nINTENT=\"Inspect the release\"\n"
        "COMPLETION_CRITERION=\"Release risks are listed\"\n"
        "END STEP\nEND PLAN"
    )


class TextStream:
    def __init__(self, *chunks):
        self.chunks = chunks
        self.request = None

    async def stream_text(self, input, *, instructions=None, tools=None):
        self.request = {"input": input, "instructions": instructions, "tools": tools}
        for chunk in self.chunks:
            yield chunk


def test_planner_decodes_plan_and_prompts_without_provider_formatting():
    stream = TextStream(plan_document())
    plan = asyncio.run(Planner(stream).plan(AgentState("Prepare release", memory_summary="Version 2.0")))

    assert plan.revision == 1
    assert plan.goal == "Prepare release"
    assert plan.steps[0].id == "inspect"
    assert stream.request["tools"] is None
    assert "BEGIN PLAN" in stream.request["instructions"]
    assert '"memory_summary": "Version 2.0"' in stream.request["input"]


def test_planner_keeps_conversation_history_separate_from_agent_goal():
    stream = TextStream(plan_document(goal="Continue"))
    conversation_input = {"history": [{"sequence": 1, "answer": "Earlier"}], "current_input": "Continue"}

    asyncio.run(Planner(stream).plan(AgentState("Continue"), conversation_input=conversation_input))

    assert json.loads(stream.request["input"])["conversation_input"] == conversation_input


@pytest.mark.parametrize(
    "document",
    [
        "not protocol",
        "BEGIN PLAN\nREVISION=1\nGOAL=\"goal\"\nEND PLAN",
        plan_document().replace('ID="inspect"', "ID=1"),
        plan_document().replace("REVISION=1", "REVISION=2"),
        plan_document().replace("END PLAN", "UNKNOWN=true\nEND PLAN"),
    ],
)
def test_planner_rejects_invalid_protocol_or_plan_contract(document):
    with pytest.raises(PlanningValidationError):
        asyncio.run(Planner(TextStream(document)).plan(AgentState("Prepare release")))


def test_planner_builds_next_revision_without_mutating_prior_plan():
    prior = Plan(1, "Prepare release", (PlanStep("one", "Inspect", "Done"),))
    plan = asyncio.run(Planner(TextStream(plan_document(revision=2))).plan(AgentState("Prepare release", plan_history=(prior,))))
    assert plan.revision == 2
    assert prior.revision == 1
