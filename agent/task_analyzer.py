"""Structured task analysis and deterministic execution-mode routing."""

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from agent.execution import ExecutionAnswer, ExecutionMode, ExecutionModeName
from llm.line_protocol import CODEC_REGISTRY, LineProtocolError, parse_line_protocol
from llm.text_stream import TextModel, validated_text
from agent.model_request import trace_llm_validation_retry


TaskType = Literal[
    "generation",
    "transformation",
    "retrieval",
    "tool_action",
    "workflow",
    "diagnosis",
    "research",
    "coding",
    "planning",
    "mixed",
]
Horizon = Literal["short", "medium", "long"]


class TaskAnalysisValidationError(ValueError):
    """The model did not return a valid task analysis."""


@dataclass(frozen=True, slots=True)
class TaskAnalysis:
    task_type: TaskType
    goal_clarity: float
    needs_tools: bool
    expected_steps: int
    expected_horizon: Horizon
    reasoning_summary: str


@dataclass(frozen=True, slots=True)
class RoutedExecutionAnswer:
    """An Execution answer together with its routing decision and evidence."""

    execution: ExecutionAnswer
    execution_mode: ExecutionModeName
    analysis: TaskAnalysis | None
    analysis_error: str | None = None


_TASK_ANALYZER_INSTRUCTIONS = """
You are a Task Analyzer in an agent runtime.

Your job is not to solve the user's task and not to choose an agent architecture directly.

Your only job is to analyze the task and produce structured task characteristics that a deterministic router can use to select the appropriate execution architecture.

Analyze the user's request based only on currently available information.

## Principles

1. Do not overestimate complexity.
2. Prefer the simplest reasonable interpretation of the task.
3. Do not assume hidden sub-tasks unless they are strongly implied by the request.
4. A long workflow is not necessarily an agentic task.
5. Do not mark a task as complex merely because it involves multiple tools.
6. Estimate task characteristics, not the implementation architecture.

## Analyze the following dimensions

### goal_clarity

How clearly defined is the final desired outcome?

Range:

* 0.0 = highly ambiguous goal
* 1.0 = clearly defined output or success condition

### needs_tools

Whether external tools, APIs, databases, files, browsers, code execution, or other environment interaction are required.

Boolean.

### expected_steps

Estimate the number of meaningful execution steps.

Use an integer.

Do not count trivial internal reasoning steps.

Examples:

* answer factual question → 1
* call one API and summarize → 2
* investigate production incident → potentially 5-15+

### expected_horizon

One of:

* "short"
* "medium"
* "long"

Guideline:

* short: usually <= 3 meaningful steps
* medium: roughly 4-8 steps
* long: usually > 8 steps or requires sustained state tracking

### task_type

Choose the closest semantic task type:

* "generation"
* "transformation"
* "retrieval"
* "tool_action"
* "workflow"
* "diagnosis"
* "research"
* "coding"
* "planning"
* "mixed"

## Important distinctions

### Plan-oriented task

If the task contains multiple dependent subgoals, competing hypotheses, or requires maintaining global progress across many steps:

Example:
"Investigate why conversion dropped over the last 3 months and propose solutions."

Then typically:

* expected_horizon = long

Do not recommend an architecture, expose chain-of-thought, or include implementation details.
"""


_TASK_ANALYSIS_INSTRUCTIONS = _TASK_ANALYZER_INSTRUCTIONS + """
## Output requirements

Return exactly one Line Protocol block:

BEGIN TASK_ANALYSIS
TASK_TYPE="research"
GOAL_CLARITY=0.0
NEEDS_TOOLS=false
EXPECTED_STEPS=1
EXPECTED_HORIZON="short"
REASONING_SUMMARY="Briefly explain the main task characteristics without recommending an architecture."
END TASK_ANALYSIS

Use uppercase snake-case field names exactly as shown. Scalar values must be JSON
literals (strings quoted and escaped). Do not include prose or additional fields.
"""


_TASK_TYPES = frozenset(TaskType.__args__)
_HORIZONS = frozenset(Horizon.__args__)
_SCORES = ("goal_clarity",)

_TASK_ANALYSIS_PROTOCOL_FIELDS = {
    "TASK_TYPE": "task_type",
    "GOAL_CLARITY": "goal_clarity",
    "NEEDS_TOOLS": "needs_tools",
    "EXPECTED_STEPS": "expected_steps",
    "EXPECTED_HORIZON": "expected_horizon",
    "REASONING_SUMMARY": "reasoning_summary",
}

class TaskAnalyzer:
    """Ask a model for descriptive task characteristics, never a mode choice."""

    def __init__(self, text_stream: TextModel, *, trace: Any | None = None, stream: bool = True) -> None:
        self._text_stream = text_stream
        self._trace = trace
        self._stream = stream

    async def run(self, goal: str, *, conversation_input: Any = None) -> TaskAnalysis:
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")
        model_input = _analysis_input(goal, conversation_input)
        return await validated_text(
            lambda instructions: self._request_analysis_text(model_input, instructions),
            instructions=_TASK_ANALYSIS_INSTRUCTIONS,
            validate=_parse_analysis,
            on_retry=lambda error: trace_llm_validation_retry(self._trace, "task_analyzer", error),
        )

    async def _request_analysis_text(self, model_input: str, instructions: str) -> str:
        if self._stream:
            return "".join(
                [
                    chunk
                    async for chunk in self._text_stream.stream_text(
                        model_input, instructions=instructions, tools=None
                    )
                ]
            )
        return await self._text_stream.complete_text(
            model_input, instructions=instructions, tools=None
        )


def _analysis_input(goal: str, conversation_input: Any = None) -> str:
    """Render the current request together with prior public conversation turns."""
    if conversation_input is None:
        return goal
    history = (
        conversation_input["history"]
        if isinstance(conversation_input, Mapping)
        else conversation_input.history
    )
    current_input = (
        conversation_input["current_input"]
        if isinstance(conversation_input, Mapping)
        else conversation_input.current_input
    )
    return json.dumps(
        {"conversation_history": list(history), "current_input": current_input},
        ensure_ascii=False,
        sort_keys=True,
    )


def route(analysis: TaskAnalysis | Mapping[str, Any]) -> ExecutionModeName:
    """Select an Execution mode using the agreed deterministic policy."""
    values: Mapping[str, Any] = asdict(analysis) if isinstance(analysis, TaskAnalysis) else analysis
    if not values["needs_tools"]:
        mode: ExecutionModeName = "direct"
    elif values["expected_steps"] <= 1:
        mode = "tool_agent"
    elif (
        values["expected_horizon"] == "long"
    ):
        mode = "plan_execute"
    else:
        mode = "react"
    return mode


class TaskRouter:
    """Analyze a goal before executing the selected whole-task Execution mode."""

    def __init__(
        self,
        analyzer: TaskAnalyzer,
        mode_factory: Callable[[ExecutionModeName], ExecutionMode],
    ) -> None:
        self._analyzer = analyzer
        self._mode_factory = mode_factory

    async def run(self, goal: str) -> RoutedExecutionAnswer:
        try:
            analysis = await self._analyzer.run(goal)
            mode = route(analysis)
            analysis_error = None
        except Exception:
            analysis = None
            mode = "plan_execute"
            analysis_error = _safe_analysis_error()
        execution = await self._mode_factory(mode).run(goal)
        return RoutedExecutionAnswer(execution, mode, analysis, analysis_error)


def _parse_analysis(output: str) -> TaskAnalysis:
    try:
        block = parse_line_protocol(output)
    except LineProtocolError as error:
        raise TaskAnalysisValidationError(str(error)) from error
    if block.type != "TASK_ANALYSIS":
        raise TaskAnalysisValidationError("expected BEGIN TASK_ANALYSIS")
    if block.children:
        raise TaskAnalysisValidationError("TASK_ANALYSIS cannot contain child blocks")
    unexpected = set(block.fields) - set(_TASK_ANALYSIS_PROTOCOL_FIELDS)
    if unexpected:
        raise TaskAnalysisValidationError("TASK_ANALYSIS contains undeclared fields")
    if set(block.fields) != set(_TASK_ANALYSIS_PROTOCOL_FIELDS):
        raise TaskAnalysisValidationError("task analysis must contain exactly the required fields")
    document: dict[str, Any] = {}
    for protocol_name, field_name in _TASK_ANALYSIS_PROTOCOL_FIELDS.items():
        values = block.fields[protocol_name]
        if len(values) != 1:
            raise TaskAnalysisValidationError(f"task analysis field {protocol_name} must occur exactly once")
        document[field_name] = values[0]
    for name in _SCORES:
        value = document[name]
        if type(value) not in {int, float} or not 0 <= value <= 1:
            raise TaskAnalysisValidationError(f"task analysis {name} must be a score from 0 to 1")
        document[name] = float(value)
    for name in ("expected_steps",):
        value = document[name]
        if type(value) is not int or value < (1 if name == "expected_steps" else 0):
            raise TaskAnalysisValidationError(f"task analysis {name} must be a non-negative integer")
    if type(document["needs_tools"]) is not bool:
        raise TaskAnalysisValidationError("task analysis needs_tools must be a boolean")
    if document["task_type"] not in _TASK_TYPES or document["expected_horizon"] not in _HORIZONS:
        raise TaskAnalysisValidationError("task analysis contains an unsupported category")
    if not isinstance(document["reasoning_summary"], str) or not document["reasoning_summary"].strip():
        raise TaskAnalysisValidationError("task analysis reasoning_summary must be non-empty text")
    return TaskAnalysis(**document)


def _safe_analysis_error() -> str:
    return "task analysis failed"
