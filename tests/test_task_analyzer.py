import asyncio

import pytest

from agent import TaskAnalysis, TaskAnalyzer
from agent.task_analyzer import TaskAnalysisValidationError


class TextStream:
    def __init__(self, *chunks: str) -> None:
        self.chunks = chunks
        self.request = None

    async def stream_text(self, input, *, instructions=None, tools=None, text_format=None):
        self.request = {
            "input": input,
            "instructions": instructions,
            "tools": tools,
            "text_format": text_format,
        }
        for chunk in self.chunks:
            yield chunk


def valid_analysis() -> str:
    return (
        '{"task_type":"research","goal_clarity":0.9,"needs_tools":true,'
        '"tool_diversity":0.4,"known_steps":0.3,"path_uncertainty":0.8,'
        '"step_dependency":0.7,"dynamic_branching":0.6,"expected_steps":5,'
        '"expected_horizon":"medium","failure_recovery":0.2,'
        '"need_replanning":0.4,"open_subgoals":2,"parallelizable":false,'
        '"risk_level":"low","reasoning_summary":"The task needs investigation."}'
    )


def test_task_analyzer_returns_the_validated_task_characteristics_contract():
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


def test_task_analyzer_requests_the_goal_with_the_strict_analysis_contract():
    text_stream = TextStream(valid_analysis())

    asyncio.run(TaskAnalyzer(text_stream).run("Investigate the report"))

    assert text_stream.request["input"] == "Investigate the report"
    assert text_stream.request["tools"] is None
    assert text_stream.request["text_format"]["type"] == "json_schema"
    assert text_stream.request["text_format"]["strict"] is True
    assert text_stream.request["text_format"]["name"] == "task_analysis"
    assert "Return JSON only" in text_stream.request["instructions"]
    assert "choose an agent architecture directly" in text_stream.request["instructions"]


def test_task_analyzer_supports_json_object_structured_output_mode():
    text_stream = TextStream(valid_analysis())

    analysis = asyncio.run(
        TaskAnalyzer(text_stream, response_format="json_object").run("Investigate the report")
    )

    assert analysis.task_type == "research"
    assert text_stream.request["text_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ("not JSON", "strict JSON"),
        ('{"task_type":"research"}', "exactly"),
        (valid_analysis().replace('"risk_level":"low"', '"risk_level":"low","extra":true'), "exactly"),
        (valid_analysis().replace('"task_type":"research"', '"task_type":"unknown"'), "unsupported category"),
        (valid_analysis().replace('"goal_clarity":0.9', '"goal_clarity":1.1'), "score"),
        (valid_analysis().replace('"expected_steps":5', '"expected_steps":0'), "non-negative integer"),
        (valid_analysis().replace('"needs_tools":true', '"needs_tools":1'), "booleans"),
        (valid_analysis().replace('"reasoning_summary":"The task needs investigation."', '"reasoning_summary":"  "'), "non-empty text"),
    ],
)
def test_task_analyzer_rejects_invalid_structured_output(output, message):
    with pytest.raises(TaskAnalysisValidationError, match=message):
        asyncio.run(TaskAnalyzer(TextStream(output)).run("Investigate the report"))


def test_task_analyzer_rejects_an_empty_goal_before_requesting_model_output():
    text_stream = TextStream(valid_analysis())

    with pytest.raises(ValueError, match="non-empty"):
        asyncio.run(TaskAnalyzer(text_stream).run("  "))

    assert text_stream.request is None


def test_task_analyzer_propagates_model_failures_without_producing_analysis():
    class FailingTextStream(TextStream):
        async def stream_text(self, *args, **kwargs):
            raise RuntimeError("provider unavailable")
            yield "unreachable"

    with pytest.raises(RuntimeError, match="provider unavailable"):
        asyncio.run(TaskAnalyzer(FailingTextStream()).run("Investigate the report"))
