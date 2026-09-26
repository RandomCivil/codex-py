import pytest
import asyncio
from langchain_core.messages import AIMessage

from agent.execution import ExecutionAnswer, ReactMode
from llm.line_protocol import LineProtocolError, decode_completion_judgment
from tests.helpers import Runtime, ToolModel


def _verdicts(*children, all_completed=False):
    return (
        "BEGIN COMPLETION_PROGRESS\n"
        f"ALL_COMPLETED={str(all_completed).lower()}\n"
        + "".join(children)
        + "END COMPLETION_PROGRESS"
    )


def _verdict(number, verified, detail):
    field = "EVIDENCE" if verified else "GAP"
    return (
        "BEGIN CRITERION_VERDICT\n"
        f"NUMBER={number}\nVERIFIED={str(verified).lower()}\n"
        f'{field}="{detail}"\nEND CRITERION_VERDICT\n'
    )


def test_completion_judgment_requires_current_verdict_for_each_ordered_criterion():
    judgment = decode_completion_judgment(
        _verdicts(
            _verdict(1, True, "The file exists."),
            _verdict(2, False, "The contents are not verified."),
        ),
        criterion_count=2,
    )

    assert judgment.verdicts == (
        (1, True, "The file exists."),
        (2, False, "The contents are not verified."),
    )
    assert judgment.all_completed is False
    assert judgment.answer is None


@pytest.mark.parametrize(
    "document",
    [
        _verdicts(_verdict(1, True, "only one"), all_completed=False),
        _verdicts(
            _verdict(1, True, "first"),
            _verdict(1, False, "duplicate"),
            all_completed=False,
        ),
        _verdicts(
            _verdict(2, True, "second"),
            _verdict(1, False, "first"),
            all_completed=False,
        ),
        _verdicts(
            _verdict(1, True, "verified"),
            _verdict(2, False, "gap"),
            all_completed=True,
        ),
        "BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=false\n"
        "BEGIN CRITERION_VERDICT\nNUMBER=1\nVERIFIED=true\nGAP=\"wrong field\"\n"
        "END CRITERION_VERDICT\nBEGIN CRITERION_VERDICT\nNUMBER=2\nVERIFIED=false\n"
        "GAP=\"gap\"\nEND CRITERION_VERDICT\nEND COMPLETION_PROGRESS",
    ],
)
def test_completion_judgment_rejects_incomplete_or_inconsistent_current_verdicts(document):
    with pytest.raises(LineProtocolError):
        decode_completion_judgment(document, criterion_count=2)


def test_positive_completion_judgment_does_not_author_the_candidate_answer():
    document = _verdicts(
        _verdict(1, True, "first"),
        _verdict(2, True, "second"),
        all_completed=True,
    )
    judgment = decode_completion_judgment(document, criterion_count=2)
    assert judgment.answer is None


def test_completion_judgment_rejects_a_judge_authored_answer():
    document = (
        "BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=true\n"
        + _verdict(1, True, "first")
        + _verdict(2, True, "second")
        + 'ANSWER="Judge answer must not escape."\nEND COMPLETION_PROGRESS'
    )

    with pytest.raises(LineProtocolError):
        decode_completion_judgment(document, criterion_count=2)


def test_react_judges_only_terminal_proposal_and_returns_candidate_answer():
    model = ToolModel(
        AIMessage(content="Inspecting", tool_calls=[{"name": "inspect", "args": {}, "id": "one"}]),
        AIMessage(content='BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="Ready."\nEND REACT_DECISION'),
        AIMessage(content=_verdicts(_verdict(1, True, "The result is present."), all_completed=True)),
    )

    answer = asyncio.run(
        ReactMode(model, Runtime({"result": "present"}), completion_criteria=("Result is present",)).run("Inspect result")
    )

    assert answer == ExecutionAnswer("Ready.", "completed")
    assert len(model.requests) == 3
    assert model.request_tools[1] != ()
    assert model.request_tools[2] == ()
    react_prompt = "\n".join(getattr(m, "content", "") for m in model.requests[1])
    assert "1. [pending] Result is present" in react_prompt


def test_react_negative_verdict_is_latest_feedback_with_criteria_text():
    model = ToolModel(
        AIMessage(content='BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="First claim."\nEND REACT_DECISION'),
        AIMessage(content=_verdicts(_verdict(1, False, "Current result is missing."), all_completed=False)),
        AIMessage(content="Fixing", tool_calls=[{"name": "inspect", "args": {}, "id": "two"}]),
        AIMessage(content='BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="Repaired claim."\nEND REACT_DECISION'),
        AIMessage(content=_verdicts(_verdict(1, True, "Current result is present."), all_completed=True)),
    )

    answer = asyncio.run(
        ReactMode(model, Runtime({"result": "present"}), completion_criteria=("Secret criterion",)).run("Inspect result")
    )

    assert answer == ExecutionAnswer("Repaired claim.", "completed")
    feedback = "\n".join(getattr(m, "content", "") for m in model.requests[2])
    assert "CRITERION_VERDICT" in feedback
    assert "Current result is missing." in feedback
    assert "1. [pending] Secret criterion" in feedback


def test_later_current_verdict_can_revoke_historically_confirmed_criterion():
    model = ToolModel(
        AIMessage(content='BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="First check."\nEND REACT_DECISION'),
        AIMessage(content=_verdicts(
            _verdict(1, True, "Criterion one was verified earlier."),
            _verdict(2, False, "Criterion two remains open."),
            all_completed=False,
        )),
        AIMessage(content="Rechecking current state", tool_calls=[{"name": "inspect", "args": {}, "id": "recheck"}]),
        AIMessage(content='BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="Second check."\nEND REACT_DECISION'),
        AIMessage(content=_verdicts(
            _verdict(1, False, "Criterion one is no longer true."),
            _verdict(2, True, "Criterion two is now verified."),
            all_completed=False,
        )),
    )

    answer = asyncio.run(
        ReactMode(
            model,
            Runtime({"result": "changed"}),
            max_rounds=3,
            completion_criteria=("Criterion one", "Criterion two"),
        ).run("Check both criteria")
    )

    assert answer == ExecutionAnswer(None, "failed", "react round budget exhausted")
    latest_judge = "\n".join(getattr(message, "content", "") for message in model.requests[-1])
    assert "Criterion one was verified earlier." in latest_judge
