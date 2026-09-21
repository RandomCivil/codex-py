import asyncio
import json

import pytest

from agent import TaskAnalysis, TaskAnalyzer, route
from agent.task_analyzer import TaskAnalysisValidationError


class TextStream:
    def __init__(self, *chunks: str) -> None:
        self.chunks = chunks
        self.request = None

    async def stream_text(self, input, *, instructions=None, tools=None):
        self.request = {
            "input": input,
            "instructions": instructions,
            "tools": tools,
        }
        for chunk in self.chunks:
            yield chunk


def valid_analysis() -> str:
    return '''BEGIN TASK_ANALYSIS
TASK_TYPE="research"
GOAL_CLARITY=0.9
NEEDS_TOOLS=true
TOOL_DIVERSITY=0.4
KNOWN_STEPS=0.3
PATH_UNCERTAINTY=0.8
STEP_DEPENDENCY=0.7
DYNAMIC_BRANCHING=0.6
EXPECTED_STEPS=5
EXPECTED_HORIZON="medium"
FAILURE_RECOVERY=0.2
NEED_REPLANNING=0.4
OPEN_SUBGOALS=2
PARALLELIZABLE=false
RISK_LEVEL="low"
REASONING_SUMMARY="The task needs investigation."
END TASK_ANALYSIS'''


def test_task_analyzer_decodes_valid_line_protocol_into_existing_analysis_contract():
    output = valid_analysis()
    text_stream = TextStream(output[:37], output[37:])

    analysis = asyncio.run(TaskAnalyzer(text_stream).run("Investigate the report"))

    assert isinstance(analysis, TaskAnalysis)
    assert analysis.task_type == "research"
    assert analysis.goal_clarity == 0.9
    assert analysis.needs_tools is True
    assert analysis.tool_diversity == 0.4
    assert analysis.known_steps == 0.3
    assert analysis.path_uncertainty == 0.8
    assert analysis.step_dependency == 0.7
    assert analysis.dynamic_branching == 0.6
    assert analysis.expected_steps == 5
    assert analysis.expected_horizon == "medium"
    assert analysis.failure_recovery == 0.2
    assert analysis.need_replanning == 0.4
    assert analysis.open_subgoals == 2
    assert analysis.parallelizable is False
    assert analysis.risk_level == "low"
    assert analysis.reasoning_summary == "The task needs investigation."
    assert route(analysis) == "react"


def test_task_analyzer_prompts_for_task_analysis_without_provider_formatting_options():
    text_stream = TextStream(valid_analysis())

    asyncio.run(TaskAnalyzer(text_stream).run("Investigate the report"))

    assert text_stream.request["input"] == "Investigate the report"
    assert text_stream.request["tools"] is None
    assert "BEGIN TASK_ANALYSIS" in text_stream.request["instructions"]


def test_task_analyzer_includes_prior_conversation_history_in_model_input():
    text_stream = TextStream(valid_analysis())

    asyncio.run(
        TaskAnalyzer(text_stream).run(
            "Continue the investigation",
            conversation_input={
                "history": [
                    {
                        "sequence": 1,
                        "user_input": "Investigate the report",
                        "answer": "The report needs follow-up.",
                    }
                ],
                "current_input": "Continue the investigation",
            },
        )
    )

    request = json.loads(text_stream.request["input"])
    assert request["current_input"] == "Continue the investigation"
    assert request["conversation_history"][0]["answer"] == "The report needs follow-up."


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (valid_analysis().replace('REASONING_SUMMARY="The task needs investigation."\n', ""), "required fields"),
        (valid_analysis().replace("END TASK_ANALYSIS", "EXTRA=true\nEND TASK_ANALYSIS"), "undeclared"),
        (valid_analysis().replace('TASK_TYPE="research"', 'TASK_TYPE="unknown"'), "unsupported category"),
        (valid_analysis().replace("GOAL_CLARITY=0.9", "GOAL_CLARITY=1.1"), "score"),
        (valid_analysis().replace("EXPECTED_STEPS=5", "EXPECTED_STEPS=0"), "non-negative integer"),
        (valid_analysis().replace("NEEDS_TOOLS=true", "NEEDS_TOOLS=1"), "booleans"),
        (valid_analysis().replace('REASONING_SUMMARY="The task needs investigation."', 'REASONING_SUMMARY="  "'), "non-empty text"),
    ],
)
def test_task_analyzer_rejects_invalid_line_protocol(output, message):
    with pytest.raises(TaskAnalysisValidationError, match=message):
        asyncio.run(TaskAnalyzer(TextStream(output)).run("Investigate the report"))


class CountingTextStream(TextStream):
    def __init__(self, *chunks: str) -> None:
        super().__init__(*chunks)
        self.calls = 0

    async def stream_text(self, *args, **kwargs):
        self.calls += 1
        async for chunk in super().stream_text(*args, **kwargs):
            yield chunk


def test_task_analyzer_does_not_retry_invalid_protocol_output():
    text_stream = CountingTextStream("")

    with pytest.raises(TaskAnalysisValidationError):
        asyncio.run(TaskAnalyzer(text_stream).run("Investigate the report"))

    assert text_stream.calls == 1
