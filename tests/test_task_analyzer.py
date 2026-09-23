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
EXPECTED_STEPS=5
EXPECTED_HORIZON="medium"
REASONING_SUMMARY="The task needs investigation."
COMPLETION_CRITERION="The report root cause is identified."
END TASK_ANALYSIS'''


def test_task_analyzer_decodes_valid_line_protocol_into_existing_analysis_contract():
    output = valid_analysis()
    text_stream = TextStream(output[:37], output[37:])

    analysis = asyncio.run(TaskAnalyzer(text_stream).run("Investigate the report"))

    assert isinstance(analysis, TaskAnalysis)
    assert analysis.task_type == "research"
    assert analysis.goal_clarity == 0.9
    assert analysis.needs_tools is True
    assert analysis.expected_steps == 5
    assert analysis.expected_horizon == "medium"
    assert analysis.reasoning_summary == "The task needs investigation."
    assert analysis.completion_criteria == ("The report root cause is identified.",)
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
        (valid_analysis().replace('COMPLETION_CRITERION="The report root cause is identified."\n', ""), "requires completion criteria"),
        (valid_analysis().replace('COMPLETION_CRITERION="The report root cause is identified."', 'COMPLETION_CRITERION="  "'), "non-empty text"),
        (valid_analysis().replace('COMPLETION_CRITERION="The report root cause is identified."', 'COMPLETION_CRITERION="Same"\nCOMPLETION_CRITERION="Same"'), "distinct"),
        (valid_analysis().replace("NEEDS_TOOLS=true", "NEEDS_TOOLS=false"), "only valid for react"),
        (valid_analysis().replace("END TASK_ANALYSIS", "EXTRA=true\nEND TASK_ANALYSIS"), "undeclared"),
        (valid_analysis().replace('TASK_TYPE="research"', 'TASK_TYPE="unknown"'), "unsupported category"),
        (valid_analysis().replace("GOAL_CLARITY=0.9", "GOAL_CLARITY=1.1"), "score"),
        (valid_analysis().replace("EXPECTED_STEPS=5", "EXPECTED_STEPS=0"), "non-negative integer"),
        (valid_analysis().replace("NEEDS_TOOLS=true", "NEEDS_TOOLS=1"), "boolean"),
        (valid_analysis().replace('REASONING_SUMMARY="The task needs investigation."', 'REASONING_SUMMARY="  "'), "non-empty text"),
    ],
)
def test_task_analyzer_rejects_invalid_line_protocol(output, message):
    with pytest.raises(TaskAnalysisValidationError, match=message):
        asyncio.run(TaskAnalyzer(TextStream(output)).run("Investigate the report"))


def test_task_analyzer_omits_completion_criteria_for_non_react_analysis():
    output = valid_analysis().replace("NEEDS_TOOLS=true", "NEEDS_TOOLS=false").replace(
        'COMPLETION_CRITERION="The report root cause is identified."\n', ""
    )
    analysis = asyncio.run(TaskAnalyzer(TextStream(output)).run("Answer the question"))

    assert analysis.completion_criteria == ()


class CountingTextStream(TextStream):
    def __init__(self, *chunks: str) -> None:
        super().__init__(*chunks)
        self.calls = 0

    async def stream_text(self, *args, **kwargs):
        self.calls += 1
        async for chunk in super().stream_text(*args, **kwargs):
            yield chunk


class SequentialTextStream(TextStream):
    def __init__(self, *responses: str) -> None:
        super().__init__()
        self.responses = iter(responses)
        self.calls = 0

    async def stream_text(self, input, *, instructions=None, tools=None):
        self.calls += 1
        self.request = {"input": input, "instructions": instructions, "tools": tools}
        yield next(self.responses)


def test_task_analyzer_retries_invalid_protocol_with_validation_feedback():
    text_stream = SequentialTextStream("", valid_analysis())

    analysis = asyncio.run(TaskAnalyzer(text_stream).run("Investigate the report"))

    assert analysis.task_type == "research"
    assert text_stream.calls == 2
    assert "Validation error: protocol response is empty" in text_stream.request["instructions"]
    assert 'Previous response (JSON-encoded): ""' in text_stream.request["instructions"]


def test_task_analyzer_rejects_a_second_invalid_protocol_output():
    with pytest.raises(TaskAnalysisValidationError):
        asyncio.run(TaskAnalyzer(SequentialTextStream("", "")).run("Investigate the report"))
