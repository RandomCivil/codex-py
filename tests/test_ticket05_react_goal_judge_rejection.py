import asyncio

from langchain_core.messages import AIMessage

from agent.execution import ExecutionAnswer, ReactMode
from agent.runtime_context import RuntimeContext
from tests.helpers import Runtime, ToolModel, react_decision


def test_react_corrects_work_after_whole_goal_judge_rejection():
    model = ToolModel(
        AIMessage(content=react_decision("completed", "ANSWER", "The file is ready.")),
        AIMessage(content=(
            'BEGIN GOAL_JUDGMENT\nCOMPLETED=false\n'
            'GAP="The requested file has not been verified."\nEND GOAL_JUDGMENT'
        )),
        AIMessage(content="Checking the requested file.", tool_calls=[
            {"name": "inspect", "args": {}, "id": "inspect-1"}
        ]),
        AIMessage(content=react_decision("completed", "ANSWER", "The file is ready and verified.")),
        AIMessage(content=(
            'BEGIN GOAL_JUDGMENT\nCOMPLETED=true\n'
            'EVIDENCE="The file inspection confirms it is ready."\nEND GOAL_JUDGMENT'
        )),
    )
    runtime = Runtime({"path": "result.txt", "exists": True})

    answer = asyncio.run(ReactMode(model, runtime, max_rounds=3).run("Create and verify result.txt"))

    assert answer == ExecutionAnswer("The file is ready and verified.", "completed")
    assert len(runtime.calls) == 1
    retry_context = "\n".join(getattr(message, "content", "") for message in model.requests[2])
    assert (
        'BEGIN GOAL_JUDGMENT\nCOMPLETED=false\n'
        'GAP="The requested file has not been verified."\nEND GOAL_JUDGMENT'
    ) in retry_context
    assert "The file is ready." not in retry_context


def test_react_keeps_only_latest_validated_goal_judgment():
    model = ToolModel(
        AIMessage(content=react_decision("completed", "ANSWER", "First claim.")),
        AIMessage(content=(
            'BEGIN GOAL_JUDGMENT\nCOMPLETED=false\nGAP="First gap."\nEND GOAL_JUDGMENT'
        )),
        AIMessage(content=react_decision("completed", "ANSWER", "Second claim.")),
        AIMessage(content=(
            'BEGIN GOAL_JUDGMENT\nCOMPLETED=false\nGAP="Second gap."\nEND GOAL_JUDGMENT'
        )),
        AIMessage(content="I will inspect now.", tool_calls=[
            {"name": "inspect", "args": {}, "id": "inspect-1"}
        ]),
        AIMessage(content=react_decision("failed", "ERROR", "Unable to verify.")),
    )

    answer = asyncio.run(ReactMode(model, Runtime(), max_rounds=4).run("Verify the result"))

    assert answer == ExecutionAnswer(None, "failed", "Unable to verify.")
    retry_context = "\n".join(getattr(message, "content", "") for message in model.requests[4])
    assert "Second gap." in retry_context
    assert "First gap." not in retry_context
    assert "First claim." not in retry_context
    assert "Second claim." not in retry_context


def test_invalid_goal_judgment_is_repaired_once_before_react_continues():
    model = ToolModel(
        AIMessage(content=react_decision("completed", "ANSWER", "Unverified claim.")),
        AIMessage(content="not a judgment"),
        AIMessage(content=(
            'BEGIN GOAL_JUDGMENT\nCOMPLETED=false\nGAP="The file is missing."\nEND GOAL_JUDGMENT'
        )),
        AIMessage(content=react_decision("failed", "ERROR", "Could not create the file.")),
    )

    answer = asyncio.run(ReactMode(model, Runtime(), max_rounds=2).run("Create a file"))

    assert answer == ExecutionAnswer(None, "failed", "Could not create the file.")
    assert model.request_tools[2] == ()
    repair_context = "\n".join(getattr(message, "content", "") for message in model.requests[2])
    assert "Validation error:" in repair_context
    assert "not a judgment" in repair_context
    react_context = "\n".join(getattr(message, "content", "") for message in model.requests[3])
    assert "Unverified claim." not in react_context
    assert "The file is missing." in react_context


def test_second_invalid_goal_judgment_does_not_accept_completion():
    model = ToolModel(
        AIMessage(content=react_decision("completed", "ANSWER", "Unverified claim.")),
        AIMessage(content="malformed judgment"),
        AIMessage(content="still malformed"),
        AIMessage(content=react_decision("failed", "ERROR", "Unable to verify.")),
    )

    answer = asyncio.run(ReactMode(model, Runtime(), max_rounds=2).run("Verify a result"))

    assert answer == ExecutionAnswer(None, "failed", "Unable to verify.")
    assert len(model.requests) == 4
    assert model.request_tools[2] == ()
    assert model.request_tools[3] != ()
    assert "validation failed" in model.requests[3][-1].content
    assert "Unverified claim." not in "\n".join(
        getattr(message, "content", "") for message in model.requests[3]
    )


def test_goal_judge_uses_the_policy_selected_evidence_window():
    class WindowPolicy:
        def record_model_use(self):
            pass

        async def maintain(self, durable_state):
            return RuntimeContext(durable_state, (), ())

        async def record_tool_round(self, *_args, **_kwargs):
            pass

    model = ToolModel(
        AIMessage(content="SECRET_RAW_RESULT", tool_calls=[
            {"name": "inspect", "args": {}, "id": "inspect-1"}
        ]),
        AIMessage(content=react_decision("completed", "ANSWER", "Current candidate.")),
        AIMessage(content=(
            'BEGIN GOAL_JUDGMENT\nCOMPLETED=false\nGAP="The current state is unproven."\nEND GOAL_JUDGMENT'
        )),
        AIMessage(content=react_decision("failed", "ERROR", "Unable to verify.")),
    )

    answer = asyncio.run(
        ReactMode(model, Runtime({"value": "SECRET_RAW_RESULT"}), context_policy=WindowPolicy(), max_rounds=3)
        .run("Verify current state")
    )

    assert answer == ExecutionAnswer(None, "failed", "Unable to verify.")
    judge_context = "\n".join(getattr(message, "content", "") for message in model.requests[2])
    assert "Current candidate." in judge_context
    assert "SECRET_RAW_RESULT" not in judge_context


def test_judge_provider_failure_fails_react_execution():
    class JudgeFailureModel(ToolModel):
        async def ainvoke(self, messages):
            if "tool-free Completion judge" in messages[0].content:
                raise RuntimeError("judge provider unavailable")
            return await super().ainvoke(messages)

    model = JudgeFailureModel(AIMessage(content=react_decision("completed", "ANSWER", "Unverified.")))

    answer = asyncio.run(ReactMode(model, Runtime()).run("Verify an outcome"))

    assert answer == ExecutionAnswer(None, "failed", "react execution failed")


def test_rejected_terminal_proposal_on_final_round_exhausts_budget():
    model = ToolModel(
        AIMessage(content=react_decision("completed", "ANSWER", "Unverified claim.")),
        AIMessage(content=(
            'BEGIN GOAL_JUDGMENT\nCOMPLETED=false\nGAP="No current evidence."\nEND GOAL_JUDGMENT'
        )),
    )

    answer = asyncio.run(ReactMode(model, Runtime(), max_rounds=1).run("Verify an outcome"))

    assert answer == ExecutionAnswer(None, "failed", "react round budget exhausted")
    assert len(model.requests) == 2
