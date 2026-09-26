import asyncio

import pytest
from langchain_core.messages import AIMessage

from agent.execution import ExecutionAnswer, ReactMode
from tests.helpers import Runtime, ToolModel, react_decision


@pytest.mark.parametrize(
    "status,field,value",
    [
        ("completed", "ANSWER", "Done"),
        ("failed", "ERROR", "Cannot continue"),
        ("need_tool", None, None),
    ],
)
def test_react_rejects_every_status_in_a_tool_response_before_execution(status, field, value):
    contradictory = react_decision(status, field, value) + "\n"
    contradictory = AIMessage(
        content=contradictory,
        tool_calls=[{"name": "write_file", "args": {"path": "unsafe"}, "id": "call-1"}],
    )
    model = ToolModel(contradictory, AIMessage(content=react_decision("failed", "ERROR", "Stopped")))
    runtime = Runtime()

    answer = asyncio.run(ReactMode(model, runtime, max_rounds=2).run("Change a file"))

    assert answer == ExecutionAnswer(None, "failed", "Stopped")
    assert runtime.calls == []
    assert "Protocol error:" in model.requests[1][-1].content
    assert "Tool calls" in model.requests[1][-1].content
    assert len(model.requests) == 2


def test_react_rejects_malformed_decision_with_tool_calls_without_recording_evidence():
    model = ToolModel(
        AIMessage(
            content='BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="Done"',
            tool_calls=[{"name": "write_file", "args": {"path": "unsafe"}, "id": "call-1"}],
        )
    )
    runtime = Runtime()

    answer = asyncio.run(ReactMode(model, runtime, max_rounds=1).run("Change a file"))

    assert answer == ExecutionAnswer(None, "failed", "react round budget exhausted")
    assert runtime.calls == []
    assert len(model.requests) == 1


def test_react_allows_corrected_tool_response_after_contradiction_and_never_judges_tool_round():
    model = ToolModel(
        AIMessage(
            content='BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="Done"\nEND REACT_DECISION',
            tool_calls=[{"name": "inspect", "args": {}, "id": "bad-call"}],
        ),
        AIMessage(
            content="I will inspect the file.",
            tool_calls=[{"name": "inspect", "args": {}, "id": "good-call"}],
        ),
        AIMessage(content=react_decision("completed", "ANSWER", "The file was inspected.")),
        AIMessage(
            content='BEGIN GOAL_JUDGMENT\nCOMPLETED=true\nEVIDENCE="Inspection output supports the answer"\nEND GOAL_JUDGMENT'
        ),
    )
    runtime = Runtime({"exists": True})

    answer = asyncio.run(ReactMode(model, runtime).run("Inspect the file"))

    assert answer == ExecutionAnswer("The file was inspected.", "completed")
    assert [call["id"] for call in runtime.calls] == ["good-call"]
    assert "Protocol error:" in model.requests[1][-1].content
    assert "Tool calls" in model.requests[1][-1].content
    assert len(model.requests) == 4
    assert model.request_tools[:3] == [[], [], []]
    assert model.request_tools[3] == ()


def test_repeated_contradictory_tool_responses_consume_round_budget_without_judging():
    contradictory = AIMessage(
        content='BEGIN REACT_DECISION\nSTATUS="need_tool"\nEND REACT_DECISION',
        tool_calls=[{"name": "inspect", "args": {}, "id": "call-1"}],
    )
    model = ToolModel(contradictory, contradictory)
    runtime = Runtime()

    answer = asyncio.run(ReactMode(model, runtime, max_rounds=2).run("Inspect"))

    assert answer == ExecutionAnswer(None, "failed", "react round budget exhausted")
    assert runtime.calls == []
    assert len(model.requests) == 2
    assert all("Protocol error:" in request[-1].content for request in model.requests[1:])
