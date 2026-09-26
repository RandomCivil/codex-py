import asyncio

from langchain_core.messages import AIMessage

from agent.execution import ExecutionAnswer, ReactMode
from agent.trace import RunTrace
from tests.helpers import Runtime, ToolModel, react_decision
import pytest


def test_zero_tool_completed_proposal_requires_whole_goal_judgment():
    model = ToolModel(
        AIMessage(content=react_decision("completed", "ANSWER", "The answer is 42.")),
        AIMessage(content='BEGIN GOAL_JUDGMENT\nCOMPLETED=true\nEVIDENCE="The explanation answers the goal."\nEND GOAL_JUDGMENT'),
    )
    runtime = Runtime()

    answer = asyncio.run(ReactMode(model, runtime).run("Explain the answer"))

    assert answer == ExecutionAnswer("The answer is 42.", "completed")
    assert runtime.calls == []
    assert len(model.requests) == 2
    assert model.request_tools[1] == ()
    judge_request = "\n".join(message.content for message in model.requests[1])
    assert "Explain the answer" in judge_request
    assert "The answer is 42." in judge_request
    assert "GOAL_JUDGMENT" in judge_request


def test_failed_decision_returns_error_without_judge():
    model = ToolModel(AIMessage(content=react_decision("failed", "ERROR", "No safe way to complete.")))
    answer = asyncio.run(ReactMode(model, Runtime()).run("Change a file"))
    assert answer == ExecutionAnswer(None, "failed", "No safe way to complete.")
    assert len(model.requests) == 1


def test_need_tool_requests_another_round_without_judge_and_exhausts_budget():
    model = ToolModel(AIMessage(content=react_decision("need_tool")), AIMessage(content=react_decision("need_tool")))
    answer = asyncio.run(ReactMode(model, Runtime(), max_rounds=2).run("Inspect a file"))
    assert answer == ExecutionAnswer(None, "failed", "react round budget exhausted")
    assert len(model.requests) == 2
    assert "Select a Tool" in model.requests[1][-1].content


@pytest.mark.parametrize("response", [
    "", "done", "BEGIN REACT_DECISION\nSTATUS=\"unknown\"\nEND REACT_DECISION",
    react_decision("completed", "ANSWER", ""),
    react_decision("failed", "ERROR", ""),
    react_decision("need_tool", "ANSWER", "extra"),
    'BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="OK"\nERROR="extra"\nEND REACT_DECISION',
    'BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="OK"',
    'BEGIN REACT_DECISION\nSTATUS=[]\nEND REACT_DECISION',
])
def test_invalid_decision_gets_protocol_feedback_within_round_budget(response):
    model = ToolModel(AIMessage(content=response), AIMessage(content=react_decision("failed", "ERROR", "Stopped")))
    answer = asyncio.run(ReactMode(model, Runtime(), max_rounds=2).run("Inspect"))
    assert answer == ExecutionAnswer(None, "failed", "Stopped")
    assert "Protocol error:" in model.requests[1][-1].content
    assert len(model.requests) == 2


def test_multiline_answer_protocol_error_explains_json_string_encoding():
    malformed = (
        'BEGIN REACT_DECISION\nSTATUS="completed"\n'
        'ANSWER="## Result\n\nA multiline answer."\nEND REACT_DECISION'
    )
    model = ToolModel(
        AIMessage(content=malformed),
        AIMessage(content=react_decision("failed", "ERROR", "Stopped")),
    )

    answer = asyncio.run(ReactMode(model, Runtime(), max_rounds=2).run("Inspect"))

    assert answer == ExecutionAnswer(None, "failed", "Stopped")
    assert "ANSWER and ERROR values must each be one JSON string literal" in model.requests[0][0].content
    feedback = model.requests[1][-1].content
    assert "ANSWER must be one JSON string literal" in feedback
    assert "literal newlines" in feedback
    assert "\\n" in feedback


def test_repeated_invalid_decisions_exhaust_round_budget_without_judge():
    model = ToolModel(AIMessage(content="done"), AIMessage(content="still done"))
    answer = asyncio.run(ReactMode(model, Runtime(), max_rounds=2).run("Inspect"))
    assert answer == ExecutionAnswer(None, "failed", "react round budget exhausted")
    assert len(model.requests) == 2
    assert model.request_tools == [[], []]


def test_tool_round_does_not_judge_until_completed_proposal_and_traces_judge():
    class Trace(RunTrace):
        def __init__(self):
            super().__init__(level="error")
            self.requests = []

        def llm_request(self, component, _messages, *, static_shape):
            self.requests.append((component, static_shape["request_kind"]))

        def llm_final(self, *_args, **_kwargs):
            pass

    model = ToolModel(
        AIMessage(content="Inspecting.", tool_calls=[{"name": "inspect", "args": {}, "id": "1"}]),
        AIMessage(content=react_decision("completed", "ANSWER", "The file exists.")),
        AIMessage(content='BEGIN GOAL_JUDGMENT\nCOMPLETED=true\nEVIDENCE="Tool showed the file."\nEND GOAL_JUDGMENT'),
    )
    runtime = Runtime({"path": "file.txt", "exists": True})
    trace = Trace()
    answer = asyncio.run(ReactMode(model, runtime, trace=trace).run("Inspect file.txt"))
    assert answer == ExecutionAnswer("The file exists.", "completed")
    assert len(runtime.calls) == 1
    assert len(model.requests) == 3
    assert model.request_tools == [[], [], ()]
    judge_request = "\n".join(message.content for message in model.requests[2])
    assert "file.txt" in judge_request
    assert "The file exists." in judge_request
    assert trace.requests == [("react", "tool_round"), ("react", "tool_round"), ("completion_judge", "completion_judge")]
