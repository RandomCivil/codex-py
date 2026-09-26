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
_BOUNDARY = re.compile(r"^(BEGIN|END)[ _]([A-Z][A-Z0-9_]*)$", re.IGNORECASE)
_SCALAR = re.compile(r"^([A-Z][A-Z0-9_]*(?:\.[A-Z][A-Z0-9_]*)*)=(.+)$", re.IGNORECASE)


@dataclass
class ParsedBlock:
    type: str
    fields: dict[str, list[Any]] = field(default_factory=dict)
    children: list["ParsedBlock"] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class CompletionJudgment:
    verdicts: tuple[tuple[int, bool, str], ...]
    all_completed: bool
    answer: str | None = None

    @property
    def completed(self) -> tuple[tuple[int, str], ...]:
        """Compatibility view of currently verified criteria."""
        return tuple((number, detail) for number, verified, detail in self.verdicts if verified)


@dataclass(frozen=True, slots=True)
class ReactDecision:
    status: str
    answer: str | None = None
    error: str | None = None


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
            kind, type_name = kind.upper(), type_name.upper()
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
            path = path.upper()
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


def decode_react_decision(document: str) -> ReactDecision:
    block = parse_line_protocol(document)
    if block.type != "REACT_DECISION" or block.children or "STATUS" not in block.fields:
        raise LineProtocolError("expected one REACT_DECISION block with STATUS")
    status = _single(block, "STATUS")
    expected = {
        "completed": {"STATUS", "ANSWER"},
        "failed": {"STATUS", "ERROR"},
        "need_tool": {"STATUS"},
    }
    if not isinstance(status, str) or status not in expected or set(block.fields) != expected[status]:
        raise LineProtocolError("REACT_DECISION fields do not match STATUS")
    if status == "need_tool":
        return ReactDecision(status)
    field_name = "ANSWER" if status == "completed" else "ERROR"
    value = _single(block, field_name)
    if not isinstance(value, str) or not value.strip():
        raise LineProtocolError(f"REACT_DECISION {field_name} must be non-empty text")
    return ReactDecision(status, answer=value if status == "completed" else None,
                         error=value if status == "failed" else None)


def decode_goal_judgment(document: str) -> tuple[bool, str]:
    block = parse_line_protocol(document)
    if block.type != "GOAL_JUDGMENT" or block.children or "COMPLETED" not in block.fields:
        raise LineProtocolError("expected one GOAL_JUDGMENT block")
    completed = _single(block, "COMPLETED")
    if type(completed) is not bool:
        raise LineProtocolError("GOAL_JUDGMENT COMPLETED must be a boolean")
    name = "EVIDENCE" if completed else "GAP"
    if set(block.fields) != {"COMPLETED", name}:
        raise LineProtocolError("GOAL_JUDGMENT fields do not match COMPLETED")
    value = _single(block, name)
    if not isinstance(value, str) or not value.strip():
        raise LineProtocolError(f"GOAL_JUDGMENT {name} must be non-empty text")
    return completed, value


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
    """Decode the legacy incremental progress contract used by older callers."""
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
        number, evidence = _single(child, "NUMBER"), _single(child, "EVIDENCE")
        if type(number) is not int or not 1 <= number <= criterion_count:
            raise LineProtocolError("completed criterion number is out of range")
        if number in completed_criteria or not isinstance(evidence, str) or not evidence.strip():
            raise LineProtocolError("completed criterion is invalid or already recorded")
        judgments.append((number, evidence.strip()))
    numbers = [number for number, _ in judgments]
    if len(numbers) != len(set(numbers)):
        raise LineProtocolError("completed criterion numbers must be distinct")
    if all_completed != (len(completed_criteria | frozenset(numbers)) == criterion_count):
        raise LineProtocolError("ALL_COMPLETED must match locally recorded criterion state")
    if "ANSWER" in block.fields and not all_completed:
        raise LineProtocolError("incomplete judgment cannot include ANSWER")
    if all_completed:
        answer = _single(block, "ANSWER") if "ANSWER" in block.fields else None
        if answer is not None and (not isinstance(answer, str) or not answer.strip()):
            raise LineProtocolError("completed judgment ANSWER must be non-empty")
    return tuple(judgments)


def decode_step_completion_progress(
    document: str, *, completed: bool = False
) -> tuple[tuple[int, str], bool]:
    """Decode the single-criterion progress contract used by a Plan step judge."""
    block = parse_line_protocol(document)
    if block.type != "STEP_COMPLETION_PROGRESS" or set(block.fields) != {"ALL_COMPLETED"} or any(
        child.type != "COMPLETED_CRITERION" for child in block.children
    ):
        raise LineProtocolError("STEP_COMPLETION_PROGRESS fields are invalid")
    all_completed = _single(block, "ALL_COMPLETED")
    if type(all_completed) is not bool:
        raise LineProtocolError("ALL_COMPLETED must be a boolean")
    if completed:
        raise LineProtocolError("Plan step completion criterion was already recorded")
    judgments: list[tuple[int, str]] = []
    for child in block.children:
        if child.children or set(child.fields) != {"NUMBER", "EVIDENCE"}:
            raise LineProtocolError("COMPLETED_CRITERION fields are invalid")
        number = _single(child, "NUMBER")
        evidence = _single(child, "EVIDENCE")
        if type(number) is not int or number != 1:
            raise LineProtocolError("Plan step completion criterion must be numbered 1")
        if not isinstance(evidence, str) or not evidence.strip():
            raise LineProtocolError("completed criterion requires non-empty evidence")
        judgments.append((number, evidence.strip()))
    if len(judgments) > 1:
        raise LineProtocolError("Plan step completion criterion must be reported at most once")
    if all_completed != bool(judgments) and not (not all_completed and not judgments):
        raise LineProtocolError("ALL_COMPLETED must match locally recorded criterion state")
    if completed and all_completed:
        raise LineProtocolError("completed Plan step criterion cannot be reported again")
    return (tuple(judgments), all_completed)


def decode_completion_judgment(
    document: str, *, criterion_count: int, completed_criteria: frozenset[int] = frozenset()
) -> CompletionJudgment:
    """Decode a complete, current-state verdict for every ordered criterion."""
    if type(criterion_count) is not int or criterion_count <= 0:
        raise LineProtocolError("criterion_count must be a positive integer")
    block = parse_line_protocol(document)
    if block.type != "COMPLETION_PROGRESS" or set(block.fields) != {"ALL_COMPLETED"} or any(
        child.type != "CRITERION_VERDICT" for child in block.children
    ):
        raise LineProtocolError("COMPLETION_PROGRESS fields are invalid")
    all_completed = _single(block, "ALL_COMPLETED")
    if type(all_completed) is not bool:
        raise LineProtocolError("ALL_COMPLETED must be a boolean")
    judgments: list[tuple[int, bool, str]] = []
    for child in block.children:
        if child.children or set(child.fields) - {"NUMBER", "VERIFIED", "EVIDENCE", "GAP"}:
            raise LineProtocolError("CRITERION_VERDICT fields are invalid")
        number = _single(child, "NUMBER")
        verified = _single(child, "VERIFIED")
        if type(number) is not int or not 1 <= number <= criterion_count:
            raise LineProtocolError("criterion number is out of range")
        if type(verified) is not bool:
            raise LineProtocolError("criterion VERIFIED must be a boolean")
        expected_detail = "EVIDENCE" if verified else "GAP"
        if set(child.fields) != {"NUMBER", "VERIFIED", expected_detail}:
            raise LineProtocolError("criterion verdict must contain exactly one evidence or gap")
        detail = _single(child, expected_detail)
        if not isinstance(detail, str) or not detail.strip():
            raise LineProtocolError("criterion verdict requires non-empty evidence or gap")
        judgments.append((number, verified, detail.strip()))
    numbers = [number for number, _, _ in judgments]
    if len(judgments) != criterion_count:
        raise LineProtocolError("criterion verdict must cover every criterion exactly once")
    if numbers != list(range(1, criterion_count + 1)):
        raise LineProtocolError("criterion verdict numbers must be ordered and unique")
    if all_completed != all(verified for _, verified, _ in judgments):
        raise LineProtocolError("ALL_COMPLETED must match locally recorded criterion state")
    return CompletionJudgment(tuple(judgments), all_completed)


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
    "decode_react_decision",
    "decode_goal_judgment",
    "decode_goal_completion",
    "decode_no_tool",
    "decode_completion_progress",
    "decode_completion_judgment",
    "decode_step_completion_progress",
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
    "REACT_DECISION": {
        "fields": frozenset({"STATUS", "ANSWER", "ERROR"}),
        "scalar_arrays": frozenset(),
        "children": frozenset(),
    },
    "GOAL_JUDGMENT": {
        "fields": frozenset({"COMPLETED", "EVIDENCE", "GAP"}),
        "scalar_arrays": frozenset(),
        "children": frozenset(),
    },
    "COMPLETION_PROGRESS": {
        "fields": frozenset({"ALL_COMPLETED", "ANSWER"}),
        "scalar_arrays": frozenset(),
        "children": frozenset({"CRITERION_VERDICT"}),
    },
    "CRITERION_VERDICT": {
        "fields": frozenset({"NUMBER", "VERIFIED", "EVIDENCE", "GAP"}),
        "scalar_arrays": frozenset(),
        "children": frozenset(),
    },
    "GOAL_COMPLETION": {
        "fields": frozenset({"ANSWER", "GOAL_SATISFIED"}),
        "scalar_arrays": frozenset(),
        "children": frozenset(),
    },
    "COMPLETION_PROGRESS": {
        "fields": frozenset({"ALL_COMPLETED", "ANSWER"}),
        "scalar_arrays": frozenset(),
        "children": frozenset({"CRITERION_VERDICT", "COMPLETED_CRITERION"}),
    },
    "STEP_COMPLETION_PROGRESS": {
        "fields": frozenset({"ALL_COMPLETED"}),
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
