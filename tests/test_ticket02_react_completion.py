import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.execution import ExecutionAnswer, ReactMode
from agent.runtime_context import RuntimeContext, RuntimeContextPolicy
from llm.line_protocol import LineProtocolError, decode_completion_progress
import pytest
from tests.helpers import Runtime, ToolModel as Model


REACT_RESULT = {"path": "README.md", "exists": True}


class MixedRuntime(Runtime):
    async def invoke(self, call):
        if call["id"] == "failed":
            raise RuntimeError("permission denied")
        return await super().invoke(call)


class ContextPolicy:
    class Context:
        def as_messages(self):
            return []

    def record_model_use(self):
        return None

    async def maintain(self, _durable_state):
        return self.Context()


def test_react_judges_successful_tool_evidence_before_accepting_answer():
    model = Model(
        AIMessage(content="Unverified model claim", tool_calls=[{"name": "inspect", "args": {}, "id": "call-1"}]),
        AIMessage(content='BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="The README exists."\nEND REACT_DECISION'),
        AIMessage(content=(
            "BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=true\n"
            "BEGIN CRITERION_VERDICT\nNUMBER=1\nVERIFIED=true\nEVIDENCE=\"README.md exists now\"\n"
            "END CRITERION_VERDICT\nEND COMPLETION_PROGRESS"
        )),
    )

    answer = asyncio.run(
        ReactMode(model, Runtime(REACT_RESULT), completion_criteria=("README exists",)).run("Inspect README")
    )

    assert answer == ExecutionAnswer("The README exists.", "completed")
    assert len(model.requests) == 3
    react_request = model.requests[1]
    assert react_request[-1].content == (
        "## Completion criteria\n\n1. [pending] README exists"
    )
    judge_request = model.requests[2]
    judge_text = "\n".join(getattr(message, "content", "") for message in judge_request)
    assert "README exists" in judge_text
    assert '"path": "README.md"' in judge_text
    assert "COMPLETION_PROGRESS" in judge_text
    assert model.request_tools[2] == ()
    assert all(isinstance(message, (SystemMessage, HumanMessage)) for message in judge_request)
    assert "The README exists." in judge_text
    assert "CRITERION_VERDICT" in judge_text
    assert "BEGIN COMPLETION_PROGRESS" in judge_text
    assert "Use Line Protocol, not Markdown" in judge_text
    assert 'EVIDENCE="The requested file exists."' in judge_text


def test_judge_preserves_runtime_context_sections_as_plain_text():
    class FormattedContextPolicy(ContextPolicy):
        async def maintain(self, durable_state):
            return RuntimeContext(durable_state, (), ())

        async def record_tool_round(self, *_args, **_kwargs):
            return None

    model = Model(
        AIMessage(content="", tool_calls=[{"name": "inspect", "args": {}, "id": "call-1"}]),
        AIMessage(content='BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="Done."\nEND REACT_DECISION'),
        AIMessage(content=(
            "BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=true\n"
            "BEGIN CRITERION_VERDICT\nNUMBER=1\nVERIFIED=true\nEVIDENCE=\"README.md exists now\"\n"
            "END CRITERION_VERDICT\nEND COMPLETION_PROGRESS"
        )),
    )

    answer = asyncio.run(
        ReactMode(
            model,
            Runtime(REACT_RESULT),
            context_policy=FormattedContextPolicy(),
            completion_criteria=("README exists",),
        ).run("Inspect README")
    )

    assert answer == ExecutionAnswer("Done.", "completed")
    runtime_prompt = model.requests[0][-1].content
    judge_prompt = model.requests[2][-1].content
    assert "### Completion criteria" in runtime_prompt
    assert "1. [pending] README exists" in runtime_prompt
    assert "README exists" in judge_prompt


def test_invalid_judgment_is_repaired_once_without_leaking_into_react_context():
    model = Model(
        AIMessage(content="", tool_calls=[{"name": "inspect", "args": {}, "id": "call-1"}]),
        AIMessage(content='BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="Done."\nEND REACT_DECISION'),
        AIMessage(content="not a protocol response"),
        AIMessage(content=(
            "BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=true\nBEGIN CRITERION_VERDICT\nNUMBER=1\nVERIFIED=true\n"
            'EVIDENCE="README.md exists now"\nEND CRITERION_VERDICT\nEND COMPLETION_PROGRESS'
        )),
    )

    answer = asyncio.run(
        ReactMode(model, Runtime(REACT_RESULT), completion_criteria=("README exists",)).run("Inspect README")
    )

    assert answer == ExecutionAnswer("Done.", "completed")
    assert len(model.requests) == 4
    repair_prompt = model.requests[3][-1].content
    assert "Validation error:" in repair_prompt
    assert "Use Line Protocol, not Markdown" in repair_prompt
    assert "BEGIN COMPLETION_PROGRESS" in repair_prompt
    assert "BEGIN CRITERION_VERDICT" in repair_prompt


def test_mixed_batch_failure_is_kept_in_current_evidence_for_terminal_judgment():
    model = Model(
        AIMessage(content="", tool_calls=[
            {"name": "inspect", "args": {}, "id": "success"},
            {"name": "inspect", "args": {}, "id": "failed"},
        ]),
        AIMessage(content='BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="Recovered."\nEND REACT_DECISION'),
        AIMessage(content=(
            "BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=true\nBEGIN CRITERION_VERDICT\nNUMBER=1\nVERIFIED=true\n"
            'EVIDENCE="README.md exists now"\nEND CRITERION_VERDICT\nEND COMPLETION_PROGRESS'
        )),
    )

    answer = asyncio.run(
        ReactMode(model, MixedRuntime(REACT_RESULT), completion_criteria=("README exists",)).run("Inspect README")
    )

    assert answer == ExecutionAnswer("Recovered.", "completed")
    judge_text = "\n".join(getattr(message, "content", "") for message in model.requests[2])
    assert '"path": "README.md"' in judge_text
    assert "permission denied" in judge_text
    assert len(model.requests) == 3


def test_premature_answer_is_rejected_and_consumes_a_react_round():
    model = Model(
        AIMessage(content='BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="Not yet."\nEND REACT_DECISION'),
        AIMessage(content=(
            'BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=false\nBEGIN CRITERION_VERDICT\n'
            'NUMBER=1\nVERIFIED=false\nGAP="Still unverified."\nEND CRITERION_VERDICT\n'
            'END COMPLETION_PROGRESS'
        )),
        AIMessage(content='BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="Still not yet."\nEND REACT_DECISION'),
        AIMessage(content=(
            'BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=false\nBEGIN CRITERION_VERDICT\n'
            'NUMBER=1\nVERIFIED=false\nGAP="Still unverified."\nEND CRITERION_VERDICT\n'
            'END COMPLETION_PROGRESS'
        )),
    )

    answer = asyncio.run(
        ReactMode(model, Runtime(REACT_RESULT), max_rounds=2, completion_criteria=("README exists",)).run("Inspect README")
    )

    assert answer == ExecutionAnswer(None, "failed", error="react round budget exhausted")
    assert "Secret criterion" not in "\n".join(getattr(m, "content", "") for m in model.requests[0])
    second_react_request = "\n".join(
        getattr(message, "content", "") for message in model.requests[2]
    )
    assert "CRITERION_VERDICT" in second_react_request
    assert "1. [pending] README exists" in second_react_request


def test_premature_answer_with_runtime_context_receives_generic_continuation_feedback():
    model = Model(
        AIMessage(content='BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="Not yet."\nEND REACT_DECISION'),
        AIMessage(content=(
            'BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=false\nBEGIN CRITERION_VERDICT\n'
            'NUMBER=1\nVERIFIED=false\nGAP="Still unverified."\nEND CRITERION_VERDICT\n'
            'END COMPLETION_PROGRESS'
        )),
        AIMessage(content='BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="Still not yet."\nEND REACT_DECISION'),
        AIMessage(content=(
            'BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=false\nBEGIN CRITERION_VERDICT\n'
            'NUMBER=1\nVERIFIED=false\nGAP="Still unverified."\nEND CRITERION_VERDICT\n'
            'END COMPLETION_PROGRESS'
        )),
    )

    answer = asyncio.run(
        ReactMode(
            model,
            Runtime(REACT_RESULT),
            max_rounds=2,
            context_policy=ContextPolicy(),
            completion_criteria=("README exists",),
        ).run("Inspect README")
    )

    assert answer == ExecutionAnswer(None, "failed", error="react round budget exhausted")
    feedback = "\n".join(getattr(m, "content", "") for m in model.requests[2])
    assert "CRITERION_VERDICT" in feedback
    assert "1. [pending] README exists" in feedback


def test_react_runtime_context_includes_criteria_in_the_model_prompt():
    model = Model(AIMessage(content='BEGIN ANSWER\nTEXT="Not yet."\nEND ANSWER'))

    answer = asyncio.run(
        ReactMode(
            model,
            Runtime(REACT_RESULT),
            max_rounds=1,
            context_policy=RuntimeContextPolicy(None),
            completion_criteria=("README exists",),
        ).run("Inspect README")
    )

    assert answer == ExecutionAnswer(None, "failed", error="react round budget exhausted")
    assert len(model.requests[0]) == 3
    runtime_prompt = model.requests[0][-1].content
    assert "### Goal" in runtime_prompt
    assert "### Completion criteria" in runtime_prompt
    assert "1. [pending] README exists" in runtime_prompt


def test_completion_progress_accepts_empty_progress_and_new_evidence():
    assert decode_completion_progress(
        "BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=false\nEND COMPLETION_PROGRESS", criterion_count=2
    ) == ()
    assert decode_completion_progress(
        "BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=false\nBEGIN COMPLETED_CRITERION\nNUMBER=2\n"
        'EVIDENCE="The second result is present"\nEND COMPLETED_CRITERION\n'
        "END COMPLETION_PROGRESS",
        criterion_count=2,
    ) == ((2, "The second result is present"),)


@pytest.mark.parametrize("document", [
    "BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=false\nBEGIN COMPLETED_CRITERION\nNUMBER=1\n"
    'EVIDENCE=""\nEND COMPLETED_CRITERION\nEND COMPLETION_PROGRESS',
    "BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=false\nBEGIN COMPLETED_CRITERION\nNUMBER=3\n"
    'EVIDENCE="out of range"\nEND COMPLETED_CRITERION\nEND COMPLETION_PROGRESS',
    "BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=false\nBEGIN COMPLETED_CRITERION\nNUMBER=1\n"
    'EVIDENCE="one"\nEND COMPLETED_CRITERION\nBEGIN COMPLETED_CRITERION\n'
    'NUMBER=1\nEVIDENCE="duplicate"\nEND COMPLETED_CRITERION\nEND COMPLETION_PROGRESS',
])
def test_completion_progress_rejects_invalid_judgments(document):
    with pytest.raises(LineProtocolError):
        decode_completion_progress(document, criterion_count=2, completed_criteria=frozenset({1}))
