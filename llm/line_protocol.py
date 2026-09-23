"""Strict, provider-neutral Line Protocol parsing and PLAN decoding."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from memory.state import Plan, PlanStep


class LineProtocolError(ValueError):
    """The response is not an unambiguous Line Protocol document."""


_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
_BOUNDARY = re.compile(r"^(BEGIN|END) ([A-Z][A-Z0-9_]*)$")
_SCALAR = re.compile(r"^([A-Z][A-Z0-9_]*(?:\.[A-Z][A-Z0-9_]*)*)=(.+)$")


@dataclass
class ParsedBlock:
    type: str
    fields: dict[str, list[Any]] = field(default_factory=dict)
    children: list["ParsedBlock"] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CompletionJudgment:
    completed: tuple[tuple[int, str], ...]
    all_completed: bool
    answer: str | None = None


def parse_line_protocol(document: str) -> ParsedBlock:
    """Parse one complete protocol block, without coercing scalar values."""
    if not isinstance(document, str):
        raise LineProtocolError("protocol response must be text")
    lines = document.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if not lines:
        raise LineProtocolError("protocol response is empty")
    stack: list[ParsedBlock] = []
    root: ParsedBlock | None = None
    for line in lines:
        if not line or line.startswith("#"):
            raise LineProtocolError("blank lines and comments are not permitted")
        boundary = _BOUNDARY.match(line)
        if boundary:
            kind, type_name = boundary.groups()
            if kind == "BEGIN":
                if not _NAME.fullmatch(type_name):
                    raise LineProtocolError("invalid block type")
                block = ParsedBlock(type_name)
                if stack:
                    stack[-1].children.append(block)
                elif root is not None:
                    raise LineProtocolError("multiple top-level blocks")
                else:
                    root = block
                stack.append(block)
            else:
                if not stack or stack[-1].type != type_name:
                    raise LineProtocolError("mismatched block boundary")
                stack.pop()
            continue
        scalar = _SCALAR.match(line)
        if scalar and stack:
            path, literal = scalar.groups()
            if literal != literal.strip():
                raise LineProtocolError(f"whitespace around JSON literal for {path}")
            try:
                value = json.loads(literal, parse_constant=_reject_json_constant)
            except (json.JSONDecodeError, ValueError) as error:
                raise LineProtocolError(f"invalid JSON literal for {path}") from error
            stack[-1].fields.setdefault(path, []).append(value)
            continue
        raise LineProtocolError("invalid line or text outside a block")
    if root is None or stack:
        raise LineProtocolError("incomplete protocol block")
    if root.type not in CODEC_REGISTRY:
        raise LineProtocolError(f"unregistered block type {root.type}")
    if root.type == "STEP":
        raise LineProtocolError("STEP is only valid inside PLAN")
    _validate_registered_fields(root)
    return root


def _reject_json_constant(value: str) -> Any:
    raise ValueError(value)


def _validate_registered_fields(block: ParsedBlock) -> None:
    """Reject fields/children not present in the explicit registry."""
    declaration = CODEC_REGISTRY[block.type]
    allowed = declaration["fields"]
    if set(block.fields) - allowed:
        raise LineProtocolError(f"{block.type} contains undeclared fields")
    if any(len(values) > 1 and path not in declaration["scalar_arrays"] for path, values in block.fields.items()):
        raise LineProtocolError(f"{block.type} contains a repeated non-array field")
    allowed_children = declaration["children"]
    if any(child.type not in allowed_children for child in block.children):
        raise LineProtocolError(f"{block.type} contains an undeclared child block")
    for child in block.children:
        _validate_registered_fields(child)


def _single(block: ParsedBlock, field: str) -> Any:
    values = block.fields.get(field)
    if values is None:
        raise LineProtocolError(f"missing declared field {field}")
    if len(values) != 1:
        raise LineProtocolError(f"repeated non-array field {field}")
    return values[0]


def decode_plan(document: str, *, goal: str, expected_revision: int) -> Plan:
    """Decode the explicitly registered PLAN/STEP contract into a Plan."""
    root = parse_line_protocol(document)
    if root.type != "PLAN":
        raise LineProtocolError("expected BEGIN PLAN")
    if set(root.fields) != {"REVISION", "GOAL"}:
        raise LineProtocolError("PLAN fields must be exactly REVISION and GOAL")
    steps = [child for child in root.children if child.type == "STEP"]
    if len(steps) != len(root.children) or not steps:
        raise LineProtocolError("PLAN must contain one or more STEP blocks")
    revision = _single(root, "REVISION")
    if type(revision) is not int or revision != expected_revision:
        raise LineProtocolError(f"plan revision must be {expected_revision}")
    response_goal = _single(root, "GOAL")
    if not isinstance(response_goal, str) or not response_goal.strip():
        raise LineProtocolError("plan goal must be a non-empty string")
    decoded_steps: list[PlanStep] = []
    for child in steps:
        if child.children or set(child.fields) != {"ID", "INTENT", "COMPLETION_CRITERION"}:
            raise LineProtocolError("STEP fields are invalid")
        values = [_single(child, key) for key in ("ID", "INTENT", "COMPLETION_CRITERION")]
        if not all(isinstance(value, str) for value in values):
            raise LineProtocolError("STEP fields must be strings")
        try:
            decoded_steps.append(PlanStep(*values))
        except (TypeError, ValueError) as error:
            raise LineProtocolError("plan contains an invalid step") from error
    try:
        return Plan(expected_revision, goal, tuple(decoded_steps))
    except (TypeError, ValueError) as error:
        raise LineProtocolError("plan violates its invariants") from error


def decode_answer(document: str) -> str:
    """Decode the public ANSWER contract into answer text."""
    block = parse_line_protocol(document)
    if block.type != "ANSWER" or block.children or set(block.fields) != {"TEXT"}:
        raise LineProtocolError("ANSWER must contain exactly one TEXT field")
    value = _single(block, "TEXT")
    if not isinstance(value, str) or not value.strip():
        raise LineProtocolError("ANSWER TEXT must be non-empty text")
    return value


def decode_no_tool(document: str) -> None:
    """Validate the empty tool-selection response."""
    block = parse_line_protocol(document)
    if block.type != "NO_TOOL" or block.fields or block.children:
        raise LineProtocolError("NO_TOOL must be an empty block")
    return None


def normalize_no_tool(document: str) -> str:
    """Normalize known provider no-tool spellings before strict validation."""
    if not isinstance(document, str):
        return document
    stripped = document.strip()
    if (
        stripped == "NO_TOOL"
        or stripped.endswith("\n_NO_TOOL_")
        or re.search(r"(?:^|\n)```no_tool\s*\n```$", stripped, flags=re.IGNORECASE)
    ):
        return "BEGIN NO_TOOL\nEND NO_TOOL"
    return document


def decode_goal_completion(document: str) -> str:
    """Decode a ReAct terminal completion proof."""
    block = parse_line_protocol(document)
    if block.type != "GOAL_COMPLETION" or block.children or set(block.fields) != {"ANSWER", "GOAL_SATISFIED"}:
        raise LineProtocolError("GOAL_COMPLETION fields are invalid")
    answer = _single(block, "ANSWER")
    satisfied = _single(block, "GOAL_SATISFIED")
    if not isinstance(answer, str) or not answer.strip() or satisfied is not True:
        raise LineProtocolError("goal completion does not prove the goal")
    return answer.strip()


def decode_completion_progress(
    document: str, *, criterion_count: int, completed_criteria: frozenset[int] = frozenset()
) -> tuple[tuple[int, str], ...]:
    """Decode newly completed ReAct criteria and their concise supporting evidence."""
    return decode_completion_judgment(
        document, criterion_count=criterion_count, completed_criteria=completed_criteria
    ).completed


def decode_completion_judgment(
    document: str, *, criterion_count: int, completed_criteria: frozenset[int] = frozenset()
) -> CompletionJudgment:
    """Decode a judge verdict and its all-criteria completion decision."""
    block = parse_line_protocol(document)
    if block.type != "COMPLETION_PROGRESS" or set(block.fields) - {"ALL_COMPLETED", "ANSWER"} or any(
        child.type != "COMPLETED_CRITERION" for child in block.children
    ):
        raise LineProtocolError("COMPLETION_PROGRESS fields are invalid")
    all_completed = _single(block, "ALL_COMPLETED")
    if type(all_completed) is not bool:
        raise LineProtocolError("ALL_COMPLETED must be a boolean")
    judgments: list[tuple[int, str]] = []
    for child in block.children:
        if child.children or set(child.fields) != {"NUMBER", "EVIDENCE"}:
            raise LineProtocolError("COMPLETED_CRITERION fields are invalid")
        number = _single(child, "NUMBER")
        evidence = _single(child, "EVIDENCE")
        if type(number) is not int or not 1 <= number <= criterion_count:
            raise LineProtocolError("completed criterion number is out of range")
        if number in completed_criteria:
            raise LineProtocolError("completed criterion was already recorded")
        if not isinstance(evidence, str) or not evidence.strip():
            raise LineProtocolError("completed criterion requires non-empty evidence")
        judgments.append((number, evidence.strip()))
    numbers = [number for number, _ in judgments]
    if len(numbers) != len(set(numbers)):
        raise LineProtocolError("completed criterion numbers must be distinct")
    recorded = completed_criteria | frozenset(numbers)
    if all_completed != (len(recorded) == criterion_count):
        raise LineProtocolError("ALL_COMPLETED must match locally recorded criterion state")
    answer = _single(block, "ANSWER") if "ANSWER" in block.fields else None
    if all_completed:
        if not isinstance(answer, str) or not answer.strip():
            raise LineProtocolError("completed judgment requires a non-empty ANSWER")
        return CompletionJudgment(tuple(judgments), True, answer.strip())
    if answer is not None:
        raise LineProtocolError("incomplete judgment cannot include ANSWER")
    return CompletionJudgment(tuple(judgments), False)


def decode_step_completion(document: str) -> dict[str, Any]:
    """Decode the locally validated Plan-step completion receipt."""
    block = parse_line_protocol(document)
    required = {"COMPLETED", "COMPLETION_CRITERION_MET", "RESULT"}
    if block.type != "STEP_COMPLETION" or block.children or not required.issubset(block.fields):
        raise LineProtocolError("STEP_COMPLETION fields are invalid")
    if set(block.fields) - required - {"FILES_READ", "FILES_MODIFIED", "OBSERVATIONS"}:
        raise LineProtocolError("STEP_COMPLETION fields are invalid")
    if _single(block, "COMPLETED") is not True or _single(block, "COMPLETION_CRITERION_MET") is not True:
        raise LineProtocolError("step completion does not satisfy its criterion")
    result = _single(block, "RESULT")
    if not isinstance(result, str) or not result.strip():
        raise LineProtocolError("STEP_COMPLETION RESULT must be non-empty text")
    arrays: dict[str, list[str]] = {}
    for name in ("FILES_READ", "FILES_MODIFIED", "OBSERVATIONS"):
        values = block.fields.get(name, [])
        if not all(isinstance(value, str) for value in values):
            raise LineProtocolError(f"{name} must contain only strings")
        arrays[name.lower()] = values
    return {"result": result.strip(), **arrays}


PLAN_INSTRUCTIONS = """Return exactly one Line Protocol block, with no prose before or after it.
Every scalar line uses exactly `FIELD=JSON_LITERAL`; do not use spaces or `:` in place of `=`.
REVISION must be the JSON integer for this revision, and GOAL must be a JSON string.
Each step is delimited by `BEGIN STEP` and `END STEP` and contains exactly ID, INTENT,
and COMPLETION_CRITERION as JSON strings.

Examples:

For a one-step goal:
BEGIN PLAN
REVISION=1
GOAL="Inspect the project"
BEGIN STEP
ID="inspect_source"
INTENT="Locate the relevant source files"
COMPLETION_CRITERION="Relevant source files are identified"
END STEP
END PLAN

For a goal with dependent steps:
BEGIN PLAN
REVISION=2
GOAL="Prepare a release"
BEGIN STEP
ID="inspect_changes"
INTENT="Review the changes included in the release"
COMPLETION_CRITERION="Release changes and risks are listed"
END STEP
BEGIN STEP
ID="write_release_notes"
INTENT="Write release notes from the reviewed changes"
COMPLETION_CRITERION="Release notes cover the listed changes and risks"
END STEP
END PLAN
"""


__all__ = [
    "CODEC_REGISTRY",
    "LineProtocolError",
    "ParsedBlock",
    "PLAN_INSTRUCTIONS",
    "decode_plan",
    "decode_answer",
    "decode_goal_completion",
    "decode_no_tool",
    "decode_completion_progress",
    "decode_completion_judgment",
    "normalize_no_tool",
    "parse_line_protocol",
    "decode_step_completion",
]

# The registry is deliberately explicit data rather than inferred from Python
# names or English pluralization.
CODEC_REGISTRY = {
    "PLAN": {
        "fields": frozenset({"REVISION", "GOAL"}),
        "scalar_arrays": frozenset(),
        "children": frozenset({"STEP"}),
    },
    "STEP": {
        "fields": frozenset({"ID", "INTENT", "COMPLETION_CRITERION"}),
        "scalar_arrays": frozenset(),
        "children": frozenset(),
    },
    "NO_TOOL": {"fields": frozenset(), "scalar_arrays": frozenset(), "children": frozenset()},
    "ANSWER": {"fields": frozenset({"TEXT"}), "scalar_arrays": frozenset(), "children": frozenset()},
    "GOAL_COMPLETION": {
        "fields": frozenset({"ANSWER", "GOAL_SATISFIED"}),
        "scalar_arrays": frozenset(),
        "children": frozenset(),
    },
    "COMPLETION_PROGRESS": {
        "fields": frozenset({"ALL_COMPLETED", "ANSWER"}),
        "scalar_arrays": frozenset(),
        "children": frozenset({"COMPLETED_CRITERION"}),
    },
    "COMPLETED_CRITERION": {
        "fields": frozenset({"NUMBER", "EVIDENCE"}),
        "scalar_arrays": frozenset(),
        "children": frozenset(),
    },
    "STEP_COMPLETION": {
        "fields": frozenset({
            "COMPLETED", "COMPLETION_CRITERION_MET", "RESULT",
            "FILES_READ", "FILES_MODIFIED", "OBSERVATIONS",
        }),
        "scalar_arrays": frozenset({"FILES_READ", "FILES_MODIFIED", "OBSERVATIONS"}),
        "children": frozenset(),
    },
    "TASK_ANALYSIS": {
        "fields": frozenset({
            "TASK_TYPE", "GOAL_CLARITY", "NEEDS_TOOLS", "EXPECTED_STEPS", "EXPECTED_HORIZON",
            "REASONING_SUMMARY", "COMPLETION_CRITERION",
        }),
        "scalar_arrays": frozenset({"COMPLETION_CRITERION"}),
        "children": frozenset(),
    },
    "OBSERVATION": {
        "fields": frozenset({
            "ROUND", "SOURCE_ROUND_START", "SOURCE_ROUND_END",
            "AFFECTS_CURRENT_DECISION", "AFFECTED_TARGETS",
        }),
        "scalar_arrays": frozenset({"AFFECTED_TARGETS"}),
        "children": frozenset({"EVIDENCE"}),
    },
    "EVIDENCE": {
        "fields": frozenset({"CATEGORY", "TEXT", "TOOL_CALL_ID"}),
        "scalar_arrays": frozenset({"TOOL_CALL_ID"}),
        "children": frozenset(),
    },
}
