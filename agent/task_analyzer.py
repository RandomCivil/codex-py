"""Structured task analysis and deterministic execution-mode routing."""

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from agent.execution import ExecutionAnswer, ExecutionMode, ExecutionModeName
from llm.response_format import ResponseFormat, require_response_format
from llm.text_stream import TextStream


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
RiskLevel = Literal["low", "medium", "high"]


class TaskAnalysisValidationError(ValueError):
    """The model did not return a valid task analysis."""


@dataclass(frozen=True, slots=True)
class TaskAnalysis:
    task_type: TaskType
    goal_clarity: float
    needs_tools: bool
    tool_diversity: float
    known_steps: float
    path_uncertainty: float
    step_dependency: float
    dynamic_branching: float
    expected_steps: int
    expected_horizon: Horizon
    failure_recovery: float
    need_replanning: float
    open_subgoals: int
    parallelizable: bool
    risk_level: RiskLevel
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
4. Distinguish between:

   * multiple steps that are known in advance
   * multiple steps that must be discovered dynamically
5. A long workflow is not necessarily an agentic task.
6. High uncertainty means the next action depends significantly on observations from previous actions.
7. `need_replanning` should only be high when execution results are likely to invalidate or materially change the current action plan.
8. Do not mark a task as complex merely because it involves multiple tools.
9. Estimate task characteristics, not the implementation architecture.

## Analyze the following dimensions

### goal_clarity

How clearly defined is the final desired outcome?

Range:

* 0.0 = highly ambiguous goal
* 1.0 = clearly defined output or success condition

### needs_tools

Whether external tools, APIs, databases, files, browsers, code execution, or other environment interaction are required.

Boolean.

### tool_diversity

How many distinct tool categories are likely required?

Range:

* 0.0 = no tools
* 0.2 = one tool/category
* 0.5 = a few related tools
* 1.0 = many heterogeneous tools or systems

### known_steps

Whether the major execution steps can be determined before execution begins.

Range:

* 0.0 = steps must mostly be discovered dynamically
* 1.0 = steps are mostly known in advance

### path_uncertainty

How uncertain is the execution path?

Range:

* 0.0 = deterministic or nearly deterministic path
* 1.0 = observations strongly determine what to do next

Examples:

* "Fetch order 123" → low
* "Run tests, then deploy if they pass" → low
* "Investigate why production latency increased" → high

### step_dependency

How strongly do later steps depend on outputs from previous steps?

Range:

* 0.0 = mostly independent
* 1.0 = strongly sequential and dependent

### dynamic_branching

How likely is the task to produce multiple possible next actions during execution?

Range:

* 0.0 = linear path
* 1.0 = many runtime branches or hypotheses

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

### failure_recovery

How likely is the task to require recovery from failed actions, invalid assumptions, unavailable resources, or unsuccessful attempts?

Range:

* 0.0 = little or no recovery needed
* 1.0 = recovery is a central part of the task

### need_replanning

How likely is execution evidence to require changing the overall approach rather than merely selecting the next obvious action?

Range:

* 0.0 = plan should remain stable
* 1.0 = substantial replanning is likely

### open_subgoals

Estimate how many independent or semi-independent subgoals are likely to need tracking simultaneously.

Use an integer.

### parallelizable

Whether meaningful sub-tasks can likely be executed independently in parallel.

Boolean.

### risk_level

Operational risk if the agent takes an incorrect action.

One of:

* "low"
* "medium"
* "high"

Examples:

* information retrieval → usually low
* editing files or deploying staging → medium
* production deployment, financial transaction, destructive action → high

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

### Known workflow vs dynamic reasoning

If steps are predictable before execution:

Example:
"Fetch the report, convert it to PDF, upload it, then notify the team."

Then:

* known_steps = high
* path_uncertainty = low
* need_replanning = low

Even though the task has several steps.

### ReAct-like task

If there is one primary goal and the agent must repeatedly inspect results to determine the next action:

Example:
"Find out why this API request is failing."

Then typically:

* path_uncertainty = high
* known_steps = low
* dynamic_branching = medium
* expected_horizon = medium

### Plan-oriented task

If the task contains multiple dependent subgoals, competing hypotheses, or requires maintaining global progress across many steps:

Example:
"Investigate why conversion dropped over the last 3 months and propose solutions."

Then typically:

* path_uncertainty = high
* open_subgoals = high
* expected_horizon = long
* need_replanning = high

Do not recommend an architecture, expose chain-of-thought, or include implementation details.
"""


_TASK_ANALYZER_JSON_OBJECT_CONTRACT = """
## Output requirements

Return JSON only.

Use exactly this schema:

{
    "task_type": "generation | transformation | retrieval | tool_action | workflow | diagnosis | research | coding | planning | mixed",
    "goal_clarity": 0.0,
    "needs_tools": false,
    "tool_diversity": 0.0,
    "known_steps": 0.0,
    "path_uncertainty": 0.0,
    "step_dependency": 0.0,
    "dynamic_branching": 0.0,
    "expected_steps": 1,
    "expected_horizon": "short | medium | long",
    "failure_recovery": 0.0,
    "need_replanning": 0.0,
    "open_subgoals": 0,
    "parallelizable": false,
    "risk_level": "low | medium | high",
    "reasoning_summary": "Briefly explain the main task characteristics without recommending an architecture."
}

Do not include:

* an architecture recommendation
* chain-of-thought
* implementation details
* markdown
* additional fields

"""


_TASK_ANALYZER_RETRY_INSTRUCTIONS = """
Your preceding response did not satisfy the task-analysis contract. Return the
complete task analysis again as JSON only, exactly as required. Include every
required field, including reasoning_summary. Do not wrap the JSON in Markdown
or a code fence.
"""


_TASK_ANALYSIS_FIELDS = {
    "task_type",
    "goal_clarity",
    "needs_tools",
    "tool_diversity",
    "known_steps",
    "path_uncertainty",
    "step_dependency",
    "dynamic_branching",
    "expected_steps",
    "expected_horizon",
    "failure_recovery",
    "need_replanning",
    "open_subgoals",
    "parallelizable",
    "risk_level",
    "reasoning_summary",
}
_TASK_TYPES = frozenset(TaskType.__args__)
_HORIZONS = frozenset(Horizon.__args__)
_RISK_LEVELS = frozenset(RiskLevel.__args__)
_SCORES = (
    "goal_clarity",
    "tool_diversity",
    "known_steps",
    "path_uncertainty",
    "step_dependency",
    "dynamic_branching",
    "failure_recovery",
    "need_replanning",
)

_TEXT_FORMAT = {
    "type": "json_schema",
    "name": "task_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(_TASK_ANALYSIS_FIELDS),
        "properties": {
            "task_type": {"type": "string", "enum": sorted(_TASK_TYPES)},
            **{name: {"type": "number", "minimum": 0, "maximum": 1} for name in _SCORES},
            "needs_tools": {"type": "boolean"},
            "parallelizable": {"type": "boolean"},
            "expected_steps": {"type": "integer", "minimum": 1},
            "open_subgoals": {"type": "integer", "minimum": 0},
            "expected_horizon": {"type": "string", "enum": sorted(_HORIZONS)},
            "risk_level": {"type": "string", "enum": sorted(_RISK_LEVELS)},
            "reasoning_summary": {"type": "string", "minLength": 1},
        },
    },
}


class TaskAnalyzer:
    """Ask a model for descriptive task characteristics, never a mode choice."""

    def __init__(self, text_stream: TextStream, *, response_format: ResponseFormat = "json_schema") -> None:
        self._text_stream = text_stream
        self._response_format = require_response_format(response_format)

    async def run(self, goal: str) -> TaskAnalysis:
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")
        text_format = _TEXT_FORMAT if self._response_format == "json_schema" else {"type": "json_object"}
        # The provider-side schema is authoritative when supported, but repeat
        # the contract in the prompt because compatible endpoints may ignore a
        # json_schema request and otherwise have no field-level guidance.
        instructions = _TASK_ANALYZER_INSTRUCTIONS + _TASK_ANALYZER_JSON_OBJECT_CONTRACT
        for attempt in range(3):
            output = "".join(
                [
                    chunk
                    async for chunk in self._text_stream.stream_text(
                        goal,
                        instructions=(
                            instructions
                            if attempt == 0
                            else f"{instructions} {_TASK_ANALYZER_RETRY_INSTRUCTIONS}"
                        ),
                        tools=None,
                        text_format=text_format,
                    )
                ]
            )
            try:
                return _parse_analysis(output)
            except TaskAnalysisValidationError:
                if attempt == 2:
                    raise
        raise AssertionError("task analysis retry loop exited unexpectedly")


def route(analysis: TaskAnalysis | Mapping[str, Any]) -> ExecutionModeName:
    """Select an Execution mode using the agreed deterministic policy."""
    values: Mapping[str, Any] = asdict(analysis) if isinstance(analysis, TaskAnalysis) else analysis
    if not values["needs_tools"]:
        mode: ExecutionModeName = "direct"
    elif values["expected_steps"] <= 2 and values["path_uncertainty"] < 0.3:
        mode = "tool_agent"
    elif (
        values["expected_horizon"] == "long"
        or values["open_subgoals"] >= 3
        or values["need_replanning"] > 0.7
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
        document = json.loads(_unfence_json(output))
    except json.JSONDecodeError as error:
        raise TaskAnalysisValidationError("task analysis output must be strict JSON") from error
    if not isinstance(document, dict) or set(document) != _TASK_ANALYSIS_FIELDS:
        raise TaskAnalysisValidationError("task analysis must contain exactly the required fields")
    for name in _SCORES:
        value = document[name]
        if type(value) not in {int, float} or not 0 <= value <= 1:
            raise TaskAnalysisValidationError(f"task analysis {name} must be a score from 0 to 1")
        document[name] = float(value)
    for name in ("expected_steps", "open_subgoals"):
        value = document[name]
        if type(value) is not int or value < (1 if name == "expected_steps" else 0):
            raise TaskAnalysisValidationError(f"task analysis {name} must be a non-negative integer")
    if type(document["needs_tools"]) is not bool or type(document["parallelizable"]) is not bool:
        raise TaskAnalysisValidationError("task analysis tool and parallelism values must be booleans")
    if document["task_type"] not in _TASK_TYPES or document["expected_horizon"] not in _HORIZONS or document["risk_level"] not in _RISK_LEVELS:
        raise TaskAnalysisValidationError("task analysis contains an unsupported category")
    if not isinstance(document["reasoning_summary"], str) or not document["reasoning_summary"].strip():
        raise TaskAnalysisValidationError("task analysis reasoning_summary must be non-empty text")
    return TaskAnalysis(**document)


def _unfence_json(output: str) -> str:
    """Remove one outer Markdown JSON fence while rejecting surrounding prose."""
    stripped = output.strip()
    if not (stripped.startswith("```") and stripped.endswith("```")):
        return stripped
    body = stripped[3:-3].lstrip()
    first_line, separator, remainder = body.partition("\n")
    if first_line.strip().lower() == "json":
        return remainder.strip()
    if not separator and not first_line.strip():
        return ""
    if not first_line.strip():
        return remainder.strip()
    return stripped


def _safe_analysis_error() -> str:
    return "task analysis failed"
