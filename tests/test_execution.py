import asyncio
from io import StringIO

from langchain_core.messages import AIMessage

from agent import PlanningValidationError
from agent.configuration import ComponentProviderConfiguration, ProviderConfiguration
from agent.execution import DirectMode, ExecutionAnswer, PlanExecuteMode, ReactMode, ToolAgentMode, create_execution_mode
from agent.trace import RunTrace
from tests.helpers import Runtime, ToolModel


class TextModel:
    def __init__(self, *responses): self.responses = iter(responses); self.requests = []
    async def stream_text(self, input, **kwargs):
        self.requests.append((input, kwargs))
        yield next(self.responses)


def test_plan_execute_reports_a_safe_actionable_provider_status():
    class ProviderFailure(RuntimeError):
        status_code = 402

    class Durable:
        async def run(self, *_args, **_kwargs):
            raise ProviderFailure("provider payload that must not reach the caller")

    answer = asyncio.run(PlanExecuteMode(Durable(), run_id="run-1").run("Inspect components"))

    assert answer == ExecutionAnswer(
        None,
        "failed",
        "plan-execute execution failed: provider returned HTTP 402 (Insufficient Balance)",
    )


def test_plan_execute_reports_the_safe_planning_validation_error():
    class Durable:
        async def run(self, *_args, **_kwargs):
            raise PlanningValidationError("invalid line or text outside a block")

    answer = asyncio.run(PlanExecuteMode(Durable(), run_id="run-1").run("Inspect components"))

    assert answer == ExecutionAnswer(
        None,
        "failed",
        "planning failed: invalid line or text outside a block",
    )


def test_direct_mode_decodes_answer_and_prompts_for_line_protocol():
    model = TextModel('BEGIN ANSWER\nTEXT="Ready"\nEND ANSWER')
    answer = asyncio.run(DirectMode(model).run("Prepare"))
    assert answer == ExecutionAnswer("Ready", "completed")
    instructions = model.requests[0][1]["instructions"]
    assert "BEGIN ANSWER" in instructions
    assert "TEXT=<JSON string literal>" in instructions
    assert 'TEXT="A concise answer"' in instructions
    assert '`{"text":"..."}`' in instructions
    assert "text_format" not in model.requests[0][1]


def test_direct_mode_rejects_invalid_answer():
    model = TextModel("Ready", 'BEGIN ANSWER\nTEXT="Ready"\nEND ANSWER')

    assert asyncio.run(DirectMode(model).run("Prepare")) == ExecutionAnswer("Ready", "completed")
    assert len(model.requests) == 2
    assert "Validation error: invalid line or text outside a block" in model.requests[1][1]["instructions"]


def test_direct_mode_rejects_a_second_invalid_answer():
    assert asyncio.run(DirectMode(TextModel("Ready", "Still ready")).run("Prepare")).status == "failed"


def test_tool_agent_prompts_for_answer_and_decodes_no_tool_response():
    model = ToolModel(AIMessage(content='BEGIN ANSWER\nTEXT="No tool needed"\nEND ANSWER'))
    answer = asyncio.run(ToolAgentMode(model, Runtime()).run("Answer"))
    assert answer == ExecutionAnswer("No tool needed", "completed")
    prompt = model.requests[0][0].content
    assert "BEGIN ANSWER" in prompt
    assert "TEXT=<JSON string literal>" in prompt
    assert 'TEXT="A concise answer"' in prompt


def test_tool_agent_retries_invalid_no_tool_response_with_validation_feedback():
    model = ToolModel(
        AIMessage(content="No tool needed"),
        AIMessage(content='BEGIN ANSWER\nTEXT="No tool needed"\nEND ANSWER'),
    )

    answer = asyncio.run(ToolAgentMode(model, Runtime()).run("Answer"))

    assert answer == ExecutionAnswer("No tool needed", "completed")
    assert len(model.requests) == 2
    prompt = model.requests[1][0].content
    assert "Validation error: invalid line or text outside a block" in prompt
    assert 'Previous response (JSON-encoded): "No tool needed"' in prompt


def test_tool_agent_executes_a_tool_call_with_accompanying_text():
    model = ToolModel(AIMessage(content='BEGIN ANSWER\nTEXT="bad"\nEND ANSWER', tool_calls=[{"name": "x", "args": {}, "id": "1"}]))
    runtime = Runtime()
    answer = asyncio.run(ToolAgentMode(model, runtime).run("Answer"))
    assert answer == ExecutionAnswer('{"value": "done"}', "completed")
    assert [{key: call[key] for key in ("name", "args", "id")} for call in runtime.calls] == [
        {"name": "x", "args": {}, "id": "1"}
    ]


def test_react_judges_no_tool_completion_proposal():
    model = ToolModel(
        AIMessage(content='BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="Done; inspected the components"\nEND REACT_DECISION'),
        AIMessage(content='BEGIN GOAL_JUDGMENT\nCOMPLETED=true\nEVIDENCE="The explanation covers the goal"\nEND GOAL_JUDGMENT'),
    )

    answer = asyncio.run(ReactMode(model, Runtime()).run("Complete"))

    assert answer == ExecutionAnswer("Done; inspected the components", "completed")
    assert len(model.requests) == 2
    assert "BEGIN NO_TOOL" not in model.requests[0][0].content
    assert "BEGIN ANSWER" not in model.requests[0][0].content
    assert "REACT_DECISION Line Protocol" in model.requests[0][0].content
    assert "GOAL_COMPLETION" not in model.requests[0][0].content
    assert "requested outcome as the fixed" in model.requests[0][0].content
    assert "completion standard" in model.requests[0][0].content
    assert "Before every tool call, check all three conditions" in model.requests[0][0].content
    assert "Do not repeat an" in model.requests[0][0].content
    assert "equivalent inspection" in model.requests[0][0].content
    assert "pending requested outcome requires modifying a file" in model.requests[0][0].content
    assert "available bound write tool" in model.requests[0][0].content
    assert "Do not make further read-only" in model.requests[0][0].content


def test_react_judges_a_plain_no_tool_answer_when_the_provider_omits_the_envelope():
    plain_answer = "### LLM 调用\n\n- `llm/llm.py`"
    model = ToolModel(
        AIMessage(content=plain_answer),
        AIMessage(content='BEGIN GOAL_JUDGMENT\nCOMPLETED=true\nEVIDENCE="The answer addresses the goal"\nEND GOAL_JUDGMENT'),
    )

    answer = asyncio.run(ReactMode(model, Runtime()).run("Describe LLM calls"))

    assert answer == ExecutionAnswer(plain_answer, "completed")
    assert len(model.requests) == 2
    assert "Candidate answer: ### LLM 调用" in model.requests[1][1].content


def test_react_rejects_an_empty_final_response_without_tool_calls():
    model = ToolModel(AIMessage(content=""))

    answer = asyncio.run(ReactMode(model, Runtime(), max_rounds=1).run("Complete"))

    assert answer == ExecutionAnswer(
        None,
        "failed",
        error="react round budget exhausted",
    )


def test_react_executes_tool_calls_with_accompanying_text():
    model = ToolModel(
        AIMessage(content="I'll inspect the source tree.", tool_calls=[{"name": "list_dir", "args": {}, "id": "1"}]),
        AIMessage(content='BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="The source tree is inspected."\nEND REACT_DECISION'),
        AIMessage(content='BEGIN GOAL_JUDGMENT\nCOMPLETED=true\nEVIDENCE="The tool output shows the tree"\nEND GOAL_JUDGMENT'),
    )
    runtime = Runtime()
    answer = asyncio.run(ReactMode(model, runtime).run("Inspect"))
    assert answer == ExecutionAnswer("The source tree is inspected.", "completed")
    assert len(model.requests) == 3
    assert all(
        "Before every tool call, check all three conditions" in request[0].content
        for request in model.requests[:2]
    )
    assert [{key: call[key] for key in ("name", "args", "id")} for call in runtime.calls] == [
        {"name": "list_dir", "args": {}, "id": "1"}
    ]


def test_react_executes_empty_text_tool_call_with_completion_criteria():
    model = ToolModel(
        AIMessage(content="", tool_calls=[{"name": "inspect", "args": {}, "id": "1"}]),
        AIMessage(content='BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="The evidence is inspected."\nEND REACT_DECISION'),
        AIMessage(content=(
            'BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=true\n'
            'BEGIN CRITERION_VERDICT\nNUMBER=1\nVERIFIED=true\nEVIDENCE="The evidence is inspected."\n'
            'END CRITERION_VERDICT\nEND COMPLETION_PROGRESS'
        )),
    )
    runtime = Runtime()

    answer = asyncio.run(
        ReactMode(
            model,
            runtime,
            completion_criteria=("Inspect the evidence.",),
        ).run("Investigate the issue")
    )

    assert answer == ExecutionAnswer("The evidence is inspected.", "completed")
    assert [{key: call[key] for key in ("name", "args", "id")} for call in runtime.calls] == [
        {"name": "inspect", "args": {}, "id": "1"}
    ]
    assert model.requests[0][-1].content == (
        "## Completion criteria\n\n"
        "1. [pending] Inspect the evidence."
    )
    assert len(model.requests) == 3


def test_react_accepts_plain_content_without_protocol_repair_with_pending_criteria():
    model = ToolModel(
        AIMessage(content="已完成。"),
        AIMessage(content=(
            'BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=true\n'
            'BEGIN CRITERION_VERDICT\nNUMBER=1\nVERIFIED=true\nEVIDENCE="Result provided"\n'
            'END CRITERION_VERDICT\nEND COMPLETION_PROGRESS'
        )),
    )

    answer = asyncio.run(
        ReactMode(
            model,
            Runtime(),
            max_rounds=2,
            completion_criteria=("提供结果。",),
        ).run("提供结果")
    )

    assert answer == ExecutionAnswer("已完成。", "completed")
    assert len(model.requests) == 2
    assert model.request_tools == [[], ()]


def test_react_ignores_unparseable_tool_call_reasoning_with_completion_criteria():
    model = ToolModel(
        AIMessage(
            content="Let me inspect the README before making the update.",
            tool_calls=[{"name": "inspect", "args": {}, "id": "1"}],
        ),
        AIMessage(content='BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="The README was inspected."\nEND REACT_DECISION'),
        AIMessage(content=(
            'BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=true\n'
            'BEGIN CRITERION_VERDICT\nNUMBER=1\nVERIFIED=true\nEVIDENCE="The README was inspected."\n'
            'END CRITERION_VERDICT\nEND COMPLETION_PROGRESS'
        )),
    )
    runtime = Runtime()

    answer = asyncio.run(
        ReactMode(
            model,
            runtime,
            completion_criteria=("Inspect the README.",),
        ).run("Update the README")
    )

    assert answer == ExecutionAnswer("The README was inspected.", "completed")
    assert [{key: call[key] for key in ("name", "args", "id")} for call in runtime.calls] == [
        {"name": "inspect", "args": {}, "id": "1"}
    ]
    assert model.requests[0][-1].content == (
        "## Completion criteria\n\n"
        "1. [pending] Inspect the README."
    )
    assert len(model.requests) == 3


def test_react_accumulates_tool_round_progress_and_requires_all_criteria_at_final():
    model = ToolModel(
        AIMessage(
            content=(
                'intermediate'
            ),
            tool_calls=[{"name": "inspect", "args": {}, "id": "1"}],
        ),
            AIMessage(content=[{"type": "text", "text": "intermediate"}], tool_calls=[{"name": "report", "args": {}, "id": "2"}]),
            AIMessage(content=(
                'BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="Both outcomes are verified."\nEND REACT_DECISION'
            )),
            AIMessage(content=(
                'BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=true\n'
                'BEGIN CRITERION_VERDICT\nNUMBER=1\nVERIFIED=true\nEVIDENCE="Evidence inspected"\nEND CRITERION_VERDICT\n'
                'BEGIN CRITERION_VERDICT\nNUMBER=2\nVERIFIED=true\nEVIDENCE="Cause reported"\nEND CRITERION_VERDICT\nEND COMPLETION_PROGRESS'
            )),
    )

    answer = asyncio.run(
        ReactMode(
            model,
            Runtime(),
            max_rounds=3,
            completion_criteria=("Inspect the evidence.", "Report the cause."),
        ).run("Investigate the issue")
    )

    assert answer == ExecutionAnswer("Both outcomes are verified.", "completed")
    assert len(model.requests) == 4
    expected_criteria = (
        "## Completion criteria\n\n"
        "1. [pending] Inspect the evidence.\n"
        "2. [pending] Report the cause."
    )
    assert all(request[-1].content == expected_criteria for request in model.requests[:3])


def test_react_updates_the_visible_criteria_status_after_a_rejected_terminal_proposal():
    model = ToolModel(
        AIMessage(content=(
            'BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="Partially done."\nEND REACT_DECISION'
        )),
        AIMessage(content=(
            'BEGIN COMPLETION_PROGRESS\nALL_COMPLETED=false\n'
            'BEGIN CRITERION_VERDICT\nNUMBER=1\nVERIFIED=true\nEVIDENCE="Evidence inspected"\nEND CRITERION_VERDICT\n'
            'BEGIN CRITERION_VERDICT\nNUMBER=2\nVERIFIED=false\nGAP="Cause is not reported"\nEND CRITERION_VERDICT\n'
            'END COMPLETION_PROGRESS'
        )),
        AIMessage(content='BEGIN REACT_DECISION\nSTATUS="failed"\nERROR="Stopped for test."\nEND REACT_DECISION'),
    )

    answer = asyncio.run(
        ReactMode(
            model,
            Runtime(),
            completion_criteria=("Inspect the evidence.", "Report the cause."),
        ).run("Investigate")
    )

    assert answer == ExecutionAnswer(None, "failed", "Stopped for test.")
    assert model.requests[2][-1].content == (
        "## Completion criteria\n\n"
        "1. [completed] Inspect the evidence.\n"
        "2. [pending] Report the cause."
    )


def test_react_rejects_progress_for_an_undeclared_criterion():
    model = ToolModel(AIMessage(content='BEGIN ANSWER\nTEXT="Done"\nEND ANSWER'))

    answer = asyncio.run(
        ReactMode(model, Runtime(), max_rounds=1, completion_criteria=("Inspect the evidence.",)).run("Investigate")
    )

    assert answer.status == "failed"
    assert answer.error == "react round budget exhausted"


def test_react_continues_after_a_final_response_leaves_criteria_pending():
    model = ToolModel(
        AIMessage(content='BEGIN ANSWER\nTEXT="The evidence is inspected."\nEND ANSWER'),
        AIMessage(content='BEGIN ANSWER\nTEXT="The cause is reported."\nEND ANSWER'),
    )

    answer = asyncio.run(
        ReactMode(
            model,
            Runtime(),
            max_rounds=2,
            completion_criteria=("Inspect the evidence.", "Report the cause."),
        ).run("Investigate")
    )

    assert answer == ExecutionAnswer(None, "failed", error="react round budget exhausted")
    assert len(model.requests) == 2
    assert "Inspect the evidence." not in model.requests[1][0].content


def test_react_fails_when_round_budget_expires_with_pending_criteria():
    model = ToolModel(AIMessage(content='BEGIN ANSWER\nTEXT="One outcome is done."\nEND ANSWER'))

    answer = asyncio.run(
        ReactMode(
            model,
            Runtime(),
            max_rounds=1,
            completion_criteria=("Inspect the evidence.", "Report the cause."),
        ).run("Investigate")
    )

    assert answer == ExecutionAnswer(None, "failed", error="react round budget exhausted")


def test_react_logs_the_original_exception_before_returning_safe_error():
    class FailingModel(ToolModel):
        async def ainvoke(self, messages):
            raise RuntimeError("provider exploded")

    output = StringIO()
    trace = RunTrace(output, level="error", run_id="run-123")

    answer = asyncio.run(ReactMode(FailingModel(), Runtime(), trace=trace).run("Inspect"))

    assert answer == ExecutionAnswer(None, "failed", error="react execution failed")
    assert "[run-123] [execution error] component=react" in output.getvalue()
    assert "RuntimeError: provider exploded" in output.getvalue()


def test_react_factory_owns_and_closes_a_fresh_http_client_per_asyncio_run(monkeypatch):
    """Interactive chat starts a new event loop for each submitted turn."""
    clients = []

    class ChatModel:
        def __init__(self, **kwargs):
            clients.append(kwargs["http_async_client"])

        def bind_tools(self, tools):
            self.tools = tools
            return self

        async def ainvoke(self, messages):
            if self.tools == ():
                return AIMessage(content='BEGIN GOAL_JUDGMENT\nCOMPLETED=true\nEVIDENCE="Done"\nEND GOAL_JUDGMENT')
            return AIMessage(content='BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="Done"\nEND REACT_DECISION')

    monkeypatch.setattr("agent.execution.ChatOpenAI", ChatModel)
    provider = ProviderConfiguration("https://provider.test/v1", "key", "model")
    configuration = ComponentProviderConfiguration(provider, provider, provider, react=provider)

    async def submit_turn():
        mode = create_execution_mode("react", configuration=configuration)
        assert await mode.run("Complete") == ExecutionAnswer("Done", "completed")

    asyncio.run(submit_turn())
    asyncio.run(submit_turn())

    assert len(clients) == 2
    assert clients[0] is not clients[1]
    assert all(client.is_closed for client in clients)
