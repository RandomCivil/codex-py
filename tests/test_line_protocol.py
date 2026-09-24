import asyncio

import pytest

from agent import Planner, PlanningValidationError
from llm.line_protocol import (
    CODEC_REGISTRY,
    LineProtocolError,
    decode_goal_completion,
    decode_no_tool,
    decode_plan,
    decode_completion_progress,
    decode_step_completion_progress,
    decode_step_completion,
    normalize_no_tool,
    parse_line_protocol,
)
from memory.state import AgentState


def test_plan_protocol_decodes_nested_steps_and_json_literals():
    document = '''BEGIN PLAN
REVISION=1
GOAL="Prepare \\"release\\""
BEGIN STEP
ID="inspect"
INTENT="Inspect the release"
COMPLETION_CRITERION="Risks are listed"
END STEP
END PLAN'''

    plan = decode_plan(document, goal="Prepare release", expected_revision=1)

    assert plan.goal == "Prepare release"
    assert plan.steps[0].id == "inspect"
    assert plan.steps[0].intent == "Inspect the release"


def test_plan_protocol_accepts_lowercase_and_underscore_boundaries():
    document = (
        'begin_plan\nrevision=1\ngoal="将功能,特性更新到readme.md"\n'
        'BEGIN_STEP\nid="check_readme_exists"\nintent="Check README"\n'
        'completion_criterion="README content is known"\nEND_STEP\nend_plan'
    )

    plan = decode_plan(document, goal="将功能,特性更新到readme.md", expected_revision=1)

    assert plan.steps[0].id == "check_readme_exists"


def test_line_protocol_still_rejects_mismatched_case_normalized_boundaries():
    with pytest.raises(LineProtocolError, match="mismatched block boundary"):
        parse_line_protocol("begin_plan\nend_step")


def test_line_protocol_rejects_duplicate_fields_across_letter_case():
    with pytest.raises(LineProtocolError, match="repeated non-array field"):
        parse_line_protocol('begin_answer\ntext="first"\nTEXT="second"\nend_answer')


def test_line_protocol_normalizes_crlf_and_rejects_unexpected_fields():
    parsed = parse_line_protocol("BEGIN NO_TOOL\r\nEND NO_TOOL\r\n")
    assert parsed.type == "NO_TOOL"

    with pytest.raises(LineProtocolError):
        parse_line_protocol("BEGIN PLAN\nREVISION=1\nUNKNOWN=true\nEND PLAN\n")


@pytest.mark.parametrize("literal, expected", [
    ('"quoted \\"text\\""', 'quoted "text"'), ("0", 0), ("-1.5", -1.5),
    ("true", True), ("false", False), ("null", None),
])
def test_parser_preserves_all_json_scalar_literals(literal, expected, monkeypatch):
    monkeypatch.setitem(CODEC_REGISTRY, "ANSWER", {
        "fields": frozenset({"VALUE"}), "scalar_arrays": frozenset(), "children": frozenset(),
    })
    block = parse_line_protocol(f"BEGIN ANSWER\nVALUE={literal}\nEND ANSWER")
    assert block.fields == {"VALUE": [expected]}


def test_parser_supports_declared_dot_paths_scalar_arrays_and_object_arrays(monkeypatch):
    monkeypatch.setitem(CODEC_REGISTRY, "ANSWER", {
        "fields": frozenset({"METADATA.SOURCE", "TAG"}),
        "scalar_arrays": frozenset({"TAG"}), "children": frozenset({"EVIDENCE"}),
    })
    document = (
        'BEGIN ANSWER\nMETADATA.SOURCE="model"\nTAG="one"\nTAG="two"\n'
        'BEGIN EVIDENCE\nCATEGORY="confirmed_facts"\nTEXT="fact"\n'
        'TOOL_CALL_ID="call-1"\nEND EVIDENCE\nEND ANSWER'
    )
    block = parse_line_protocol(document)
    assert block.fields == {"METADATA.SOURCE": ["model"], "TAG": ["one", "two"]}
    assert block.children[0].type == "EVIDENCE"


def test_parser_rejects_repeated_non_array_and_unregistered_child_blocks():
    with pytest.raises(LineProtocolError):
        parse_line_protocol('BEGIN ANSWER\nTEXT="one"\nTEXT="two"\nEND ANSWER')
    with pytest.raises(LineProtocolError):
        parse_line_protocol('BEGIN ANSWER\nBEGIN STEP\nID="one"\nINTENT="do"\nCOMPLETION_CRITERION="done"\nEND STEP\nEND ANSWER')


def test_decodes_goal_completion_and_step_completion_contracts():
    goal = decode_goal_completion(
        'BEGIN GOAL_COMPLETION\nANSWER="Done."\nGOAL_SATISFIED=true\nEND GOAL_COMPLETION'
    )
    assert goal == "Done."

    completion = decode_step_completion(
        'BEGIN STEP_COMPLETION\n'
        'COMPLETED=true\nCOMPLETION_CRITERION_MET=true\nRESULT="Published."\n'
        'FILES_READ="README.md"\nFILES_MODIFIED="src/app.py"\n'
        'OBSERVATIONS="Release was published."\nEND STEP_COMPLETION'
    )
    assert completion == {
        "result": "Published.",
        "files_read": ["README.md"],
        "files_modified": ["src/app.py"],
        "observations": ["Release was published."],
    }


def test_step_completion_allows_empty_declared_scalar_arrays():
    completion = decode_step_completion(
        'BEGIN STEP_COMPLETION\nCOMPLETED=true\nCOMPLETION_CRITERION_MET=true\n'
        'RESULT="Done"\nEND STEP_COMPLETION'
    )
    assert completion["files_read"] == completion["files_modified"] == completion["observations"] == []


def test_no_tool_requires_an_empty_registered_block():
    assert decode_no_tool("BEGIN NO_TOOL\nEND NO_TOOL") is None
    assert normalize_no_tool("NO_TOOL") == "BEGIN NO_TOOL\nEND NO_TOOL"
    assert normalize_no_tool("answer\n\n_NO_TOOL_") == "BEGIN NO_TOOL\nEND NO_TOOL"
    with pytest.raises(LineProtocolError):
        decode_no_tool('BEGIN NO_TOOL\nTEXT="no"\nEND NO_TOOL')


def test_completion_progress_decodes_empty_and_evidenced_judgments():
    assert decode_completion_progress(
        "BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=false\nEND COMPLETION_PROGRESS", criterion_count=2
    ) == ()
    assert decode_completion_progress(
        "BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=false\nBEGIN COMPLETED_CRITERION\nNUMBER=1\n"
        'EVIDENCE="The file exists"\nEND COMPLETED_CRITERION\nEND COMPLETION_PROGRESS',
        criterion_count=2,
    ) == ((1, "The file exists"),)


@pytest.mark.parametrize("document", [
    ('BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=false\nBEGIN COMPLETED_CRITERION\nNUMBER=1\nEVIDENCE=""\nEND COMPLETED_CRITERION\nEND COMPLETION_PROGRESS',),
    ('BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=false\nBEGIN COMPLETED_CRITERION\nNUMBER=3\nEVIDENCE="out"\nEND COMPLETED_CRITERION\nEND COMPLETION_PROGRESS',),
    ('BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=false\nBEGIN COMPLETED_CRITERION\nNUMBER=1\nEVIDENCE="one"\nEND COMPLETED_CRITERION\nBEGIN COMPLETED_CRITERION\nNUMBER=1\nEVIDENCE="two"\nEND COMPLETED_CRITERION\nEND COMPLETION_PROGRESS',),
])
def test_completion_progress_rejects_invalid_contracts(document):
    with pytest.raises(LineProtocolError):
        decode_completion_progress(document, criterion_count=2)


def test_planner_decodes_plan_protocol_without_provider_format():
    class Completion:
        async def complete_text(self, input, *, instructions=None, tools=None):
            self.request = {"instructions": instructions}
            return 'BEGIN PLAN\nREVISION=1\nGOAL="Prepare release"\nBEGIN STEP\nID="one"\nINTENT="Do one"\nCOMPLETION_CRITERION="One is done"\nEND STEP\nEND PLAN\n'

    stream = Completion()
    plan = asyncio.run(Planner(stream).plan(AgentState("Prepare release")))

    assert plan.steps[0].id == "one"
    assert "BEGIN PLAN" in stream.request["instructions"]


def test_planner_rejects_malformed_protocol():
    class Completion:
        async def complete_text(self, input, *, instructions=None, tools=None):
            return "BEGIN PLAN\nREVISION=1\nEND PLAN\n"

    with pytest.raises(PlanningValidationError):
        asyncio.run(Planner(Completion()).plan(AgentState("Prepare release")))


def test_step_completion_progress_accepts_pending_without_an_answer():
    document = "BEGIN STEP_COMPLETION_PROGRESS\nALL_COMPLETED=false\nEND STEP_COMPLETION_PROGRESS"

    assert decode_step_completion_progress(document) == ((), False)


def test_step_completion_progress_accepts_only_criterion_one_with_evidence():
    document = (
        "BEGIN STEP_COMPLETION_PROGRESS\nALL_COMPLETED=true\n"
        "BEGIN COMPLETED_CRITERION\nNUMBER=1\nEVIDENCE=\"artifact exists\"\n"
        "END COMPLETED_CRITERION\nEND STEP_COMPLETION_PROGRESS"
    )

    assert decode_step_completion_progress(document) == (((1, "artifact exists"),), True)


def test_step_completion_progress_rejects_final_answer():
    document = (
        "BEGIN STEP_COMPLETION_PROGRESS\nALL_COMPLETED=true\nANSWER=\"done\"\n"
        "BEGIN COMPLETED_CRITERION\nNUMBER=1\nEVIDENCE=\"artifact exists\"\n"
        "END COMPLETED_CRITERION\nEND STEP_COMPLETION_PROGRESS"
    )

    with pytest.raises(LineProtocolError):
        decode_step_completion_progress(document)
