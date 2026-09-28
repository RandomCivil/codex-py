import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.execution import ExecutionAnswer, ReactMode
from agent.conversation import ConversationInput
from agent.runtime_context import RuntimeContextPolicy
from tests.helpers import Runtime, ToolModel


def _decision(answer):
    return (
        'BEGIN REACT_DECISION\nSTATUS="completed"\n'
        f'ANSWER="{answer}"\nEND REACT_DECISION'
    )


def _rejected_verdict():
    return (
        "BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=false\n"
        "BEGIN CRITERION_VERDICT\nNUMBER=1\nVERIFIED=false\n"
        'GAP="The README has not been inspected."\n'
        "END CRITERION_VERDICT\nEND COMPLETION_PROGRESS"
    )


def test_react_messages_layout_places_rejected_completion_feedback_before_goal():
    model = ToolModel(
        AIMessage(content=_decision("Not yet.")),
        AIMessage(content=_rejected_verdict()),
        AIMessage(content=_decision("Still not yet.")),
        AIMessage(content=_rejected_verdict()),
    )

    answer = asyncio.run(
        ReactMode(
            model,
            Runtime(),
            max_rounds=2,
            context_policy=RuntimeContextPolicy(None, layout="messages"),
            completion_criteria=("README exists",),
        ).run("Inspect README")
    )

    assert answer == ExecutionAnswer(None, "failed", "react round budget exhausted")
    second_operational_request = model.requests[2]
    assert [type(message) for message in second_operational_request] == [
        SystemMessage,
        HumanMessage,
        HumanMessage,
        HumanMessage,
        HumanMessage,
        HumanMessage,
        HumanMessage,
    ]
    assert "## Durable state" in second_operational_request[2].content
    assert "## Observations" in second_operational_request[3].content
    assert "COMPLETION_PROGRESS" in second_operational_request[4].content
    assert "## Goal" in second_operational_request[5].content
    assert "## Completion criteria" in second_operational_request[6].content
    assert "1. [pending] README exists" in second_operational_request[6].content


def test_react_messages_layout_uses_native_tool_evidence_for_completion_judgment():
    model = ToolModel(
        AIMessage(content="", tool_calls=[
            {"name": "read_file", "args": {"path": "README.md"}, "id": "read-1"}
        ]),
        AIMessage(content=_decision("README is present.")),
        AIMessage(content=(
            "BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=true\n"
            "BEGIN CRITERION_VERDICT\nNUMBER=1\nVERIFIED=true\n"
            'EVIDENCE="README.md was read."\n'
            "END CRITERION_VERDICT\nEND COMPLETION_PROGRESS"
        )),
    )

    answer = asyncio.run(
        ReactMode(
            model,
            Runtime({"path": "README.md", "exists": True}),
            context_policy=RuntimeContextPolicy(None, layout="messages"),
            completion_criteria=("README exists",),
        ).run("Inspect README")
    )

    assert answer == ExecutionAnswer("README is present.", "completed")
    judge_request = model.requests[2]
    assert [type(message) for message in judge_request] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        ToolMessage,
        HumanMessage,
        HumanMessage,
        HumanMessage,
        HumanMessage,
        HumanMessage,
    ]
    assert judge_request[1].content == "Inspect README"
    assert judge_request[2].tool_calls[0]["id"] == "read-1"
    assert judge_request[3].tool_call_id == "read-1"
    assert "## Durable state" in judge_request[4].content
    assert "## Observations" in judge_request[5].content
    assert "Candidate answer" in judge_request[6].content
    assert "## Goal" in judge_request[7].content
    assert "## Completion criteria" in judge_request[8].content


def test_react_messages_layout_preserves_structured_conversation_input_for_goal_judgment():
    model = ToolModel(
        AIMessage(content=_decision("The project is ready.")),
        AIMessage(content=(
            "BEGIN GOAL_JUDGMENT\nCOMPLETED=true\n"
            'EVIDENCE="The response addresses the current input."\nEND GOAL_JUDGMENT'
        )),
    )
    conversation_input = ConversationInput(
        "conv-1",
        ({"user": "Earlier request", "answer": "Earlier answer"},),
        "What is ready?",
    )

    answer = asyncio.run(
        ReactMode(
            model,
            Runtime(),
            context_policy=RuntimeContextPolicy(None, layout="messages"),
        ).run(conversation_input)
    )

    assert answer == ExecutionAnswer("The project is ready.", "completed")
    judge_request = model.requests[1]
    assert [type(message) for message in judge_request] == [
        SystemMessage,
        HumanMessage,
        HumanMessage,
        HumanMessage,
        HumanMessage,
        HumanMessage,
    ]
    assert judge_request[1].content == (
        '{"history":[{"user":"Earlier request","answer":"Earlier answer"}],'
        '"current_input":"What is ready?"}'
    )
    assert "## Durable state" in judge_request[2].content
    assert "## Observations" in judge_request[3].content
    assert "Candidate answer" in judge_request[4].content
    assert "## Goal" in judge_request[5].content


def test_react_messages_layout_keeps_a_failed_call_in_its_native_pair_only():
    class FailingRuntime(Runtime):
        async def invoke(self, _call):
            raise RuntimeError("permission denied")

    model = ToolModel(
        AIMessage(content="", tool_calls=[
            {"name": "read_file", "args": {"path": "README.md"}, "id": "read-1"}
        ]),
        AIMessage(content=_decision("The read was denied.")),
        AIMessage(content=(
            "BEGIN GOAL_JUDGMENT\nCOMPLETED=true\n"
            'EVIDENCE="The error is reported."\nEND GOAL_JUDGMENT'
        )),
    )

    answer = asyncio.run(
        ReactMode(
            model,
            FailingRuntime(),
            context_policy=RuntimeContextPolicy(None, layout="messages"),
        ).run("Inspect README")
    )

    assert answer == ExecutionAnswer("The read was denied.", "completed")
    second_operational_request = model.requests[1]
    assert [type(message) for message in second_operational_request] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        ToolMessage,
        HumanMessage,
        HumanMessage,
        HumanMessage,
    ]
    assert second_operational_request[3].status == "error"
    assert "permission denied" in second_operational_request[3].content
    assert "## Goal" in second_operational_request[-1].content
