import asyncio
import json

from langchain_core.messages import AIMessage

from agent import ExecutionAnswer, create_execution_mode
from agent.configuration import load_configuration


class ControlledTextModel:
    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    async def stream_text(self, input, **kwargs):
        self.calls.append((input, kwargs))
        for chunk in self.chunks:
            yield chunk


class FailingTextModel:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    async def stream_text(self, input, **kwargs):
        self.calls += 1
        raise self.error
        yield  # keep this an async generator


class ControlledToolModel:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    async def ainvoke(self, input):
        self.calls.append(input)
        return self.response


class StructuredToolModel(ControlledToolModel):
    def __init__(self, response):
        super().__init__(response)
        self.response_format_bindings = []

    def bind(self, **kwargs):
        self.response_format_bindings.append(kwargs)
        return self


class TerminalStructuredModel:
    def __init__(self):
        self.operational_calls = []
        self.completion_calls = []
        self.operational_response_formats = []
        self.response_format_bindings = []
        self._operational_responses = iter(
            [
                AIMessage(content="", tool_calls=[{"name": "first", "args": {}, "id": "call-1"}]),
                AIMessage(content="The tool result is sufficient."),
            ]
        )

    def bind_tools(self, tools, *, response_format=None):
        self.tools = tools
        self.operational_response_formats.append(response_format)
        return self

    async def ainvoke(self, input):
        self.operational_calls.append(list(input))
        return next(self._operational_responses)

    def bind(self, **kwargs):
        self.response_format_bindings.append(kwargs)
        return TerminalCompletionModel(self)


class TerminalCompletionModel:
    def __init__(self, parent):
        self.parent = parent

    async def ainvoke(self, input):
        self.parent.completion_calls.append(list(input))
        return AIMessage(content=json.dumps({"answer": "Done.", "goal_satisfied": True}))


class SequencedToolModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    async def ainvoke(self, input):
        self.calls.append(list(input))
        return next(self.responses)


class ToolRuntime:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    @property
    def tools(self):
        return ["first", "second"]

    async def invoke(self, call):
        self.calls.append(call)
        return self.result


class CorrectingToolRuntime(ToolRuntime):
    async def invoke(self, call):
        self.calls.append(call)
        if len(self.calls) == 1:
            raise ValueError("invalid arguments")
        return self.result


class BatchToolRuntime(ToolRuntime):
    async def invoke(self, call):
        self.calls.append(call)
        await asyncio.sleep(0.02 if call["id"] == "slow" else 0)
        return {"id": call["id"]}


def test_tool_agent_executes_only_the_first_call_and_renders_its_result():
    model = ControlledToolModel(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "first", "args": {"value": 1}, "id": "call-1"},
                {"name": "second", "args": {"value": 2}, "id": "call-2"},
            ],
        )
    )
    runtime = ToolRuntime({"value": "done", "items": [2, 1]})

    async def run():
        return await create_execution_mode("tool_agent", model=model, tool_runtime=runtime).run("Do it")

    answer = asyncio.run(run())

    assert answer == ExecutionAnswer(json.dumps({"items": [2, 1], "value": "done"}, sort_keys=True), "completed")
    assert len(model.calls) == 1
    assert runtime.calls == [model.response.tool_calls[0]]
    assert runtime.closed is True


def test_tool_agent_returns_model_text_without_opening_tool_runtime_when_no_tool_is_selected():
    model = ControlledToolModel(AIMessage(content="Nothing else is needed."))
    runtime = ToolRuntime("unused")

    async def run():
        return await create_execution_mode("tool_agent", model=model, tool_runtime=runtime).run("Answer")

    answer = asyncio.run(run())

    assert answer == ExecutionAnswer("Nothing else is needed.", "completed")
    assert runtime.closed is True


class DurableResult:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def run(self, run_id, state):
        self.calls.append((run_id, state))
        return self.result


def test_plan_execute_maps_last_completed_step_handoff_to_common_answer():
    durable = DurableResult(
        {
            "status": "completed",
            "state": {
                "goal": "Ship it",
                "plan_history": [
                    {
                        "revision": 1,
                        "goal": "Ship it",
                        "steps": [{"id": "publish", "intent": "Publish", "completion_criterion": "Published"}],
                    }
                ],
                "step_executions": [
                    {"revision": 1, "step_id": "publish", "status": "completed", "result": "Published.", "error": None}
                ],
                "memory_summary": None,
            },
        }
    )

    async def run():
        return await create_execution_mode("plan_execute", durable_agent=durable).run("Ship it")

    answer = asyncio.run(run())

    assert answer == ExecutionAnswer("Published.", "completed")
    assert durable.calls[0][1].goal == "Ship it"


def test_plan_execute_maps_blocked_run_to_failed_answer_without_synthesis():
    durable = DurableResult({"status": "blocked", "state": None})

    async def run():
        return await create_execution_mode("plan_execute", durable_agent=durable).run("Ship it")

    answer = asyncio.run(run())

    assert answer.status == "failed"
    assert answer.answer is None
    assert answer.error == "plan-execute run did not complete"


def test_factory_selects_direct_mode_and_returns_one_completed_execution_answer():
    model = ControlledTextModel(["The ", "answer is ready."])

    async def run():
        runner = create_execution_mode("direct", model=model)
        return await runner.run("Prepare the release notes")

    answer = asyncio.run(run())

    assert answer == ExecutionAnswer("The answer is ready.", "completed")
    assert model.calls == [("Prepare the release notes", {"tools": None})]


def test_react_completes_only_from_a_structured_goal_satisfied_response():
    model = ControlledToolModel(
        AIMessage(
            content=json.dumps({"answer": "The release is ready.", "goal_satisfied": True}),
        )
    )
    runtime = ToolRuntime("unused")

    async def run():
        return await create_execution_mode("react", model=model, tool_runtime=runtime).run("Prepare the release")

    answer = asyncio.run(run())

    assert answer == ExecutionAnswer("The release is ready.", "completed")
    assert len(model.calls) == 1
    assert runtime.closed is True


def test_react_prepends_a_few_shot_prompt_to_the_goal():
    model = ControlledToolModel(
        AIMessage(content=json.dumps({"answer": "Done.", "goal_satisfied": True}))
    )
    runtime = ToolRuntime("unused")

    async def run():
        return await create_execution_mode("react", model=model, tool_runtime=runtime).run("Do it")

    answer = asyncio.run(run())

    assert answer == ExecutionAnswer("Done.", "completed")
    prompt, goal = model.calls[0][:2]
    assert prompt.type == "system"
    assert "Examples:" in prompt.content
    assert "call the available file-reading tool" in prompt.content
    assert goal.content == "Do it"


def test_react_uses_configured_structured_output_mode_for_its_final_response(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        """
model:
  base_url: https://shared.example/v1
  api_key: shared-secret
  model_name: shared-model
  response_format: json_object
"""
    )
    configuration = load_configuration(path)
    model = StructuredToolModel(
        AIMessage(content=json.dumps({"answer": "Done.", "goal_satisfied": True}))
    )
    runtime = ToolRuntime("unused")

    async def run():
        return await create_execution_mode(
            "react", model=model, tool_runtime=runtime, configuration=configuration
        ).run("Do it")

    answer = asyncio.run(run())

    assert answer == ExecutionAnswer("Done.", "completed")
    assert model.response_format_bindings == [{"response_format": {"type": "json_object"}}]


def test_react_applies_structured_output_only_to_the_terminal_completion_request(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        """
model:
  base_url: https://shared.example/v1
  api_key: shared-secret
  model_name: shared-model
  response_format: json_schema
"""
    )
    model = TerminalStructuredModel()
    runtime = ToolRuntime("observed")

    async def run():
        return await create_execution_mode(
            "react",
            model=model,
            tool_runtime=runtime,
            configuration=load_configuration(path),
        ).run("Do it")

    answer = asyncio.run(run())

    assert answer == ExecutionAnswer("Done.", "completed")
    assert len(model.operational_calls) == 2
    assert len(model.completion_calls) == 1
    assert model.operational_response_formats == [None]
    response_format = model.response_format_bindings[0]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "react_goal_completion"


def test_react_returns_tool_errors_to_the_model_for_a_corrective_round():
    model = SequencedToolModel(
        [
            AIMessage(content="", tool_calls=[{"name": "first", "args": {}, "id": "bad-call"}]),
            AIMessage(
                content="",
                tool_calls=[{"name": "first", "args": {"fixed": True}, "id": "good-call"}],
            ),
            AIMessage(content=json.dumps({"answer": "Corrected.", "goal_satisfied": True})),
        ]
    )
    runtime = CorrectingToolRuntime("accepted")

    async def run():
        return await create_execution_mode("react", model=model, tool_runtime=runtime).run("Fix it")

    answer = asyncio.run(run())

    assert answer == ExecutionAnswer("Corrected.", "completed")
    assert [call["id"] for call in runtime.calls] == ["bad-call", "good-call"]
    assert len(model.calls) == 3


def test_react_executes_tool_batches_concurrently_and_returns_results_in_request_order():
    model = SequencedToolModel(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "first", "args": {}, "id": "slow"},
                    {"name": "second", "args": {}, "id": "fast"},
                ],
            ),
            AIMessage(content=json.dumps({"answer": "Both observed.", "goal_satisfied": True})),
        ]
    )
    runtime = BatchToolRuntime("unused")

    async def run():
        return await create_execution_mode("react", model=model, tool_runtime=runtime).run("Observe both")

    answer = asyncio.run(run())

    assert answer == ExecutionAnswer("Both observed.", "completed")
    tool_messages = model.calls[1][-2:]
    assert [message.tool_call_id for message in tool_messages] == ["slow", "fast"]
    assert [message.content for message in tool_messages] == [
        '{"id": "slow"}',
        '{"id": "fast"}',
    ]


def test_react_rejects_a_final_response_without_goal_satisfaction_proof():
    model = ControlledToolModel(
        AIMessage(content=json.dumps({"answer": "Probably done.", "goal_satisfied": False}))
    )
    runtime = ToolRuntime("unused")

    async def run():
        return await create_execution_mode("react", model=model, tool_runtime=runtime).run("Do it")

    answer = asyncio.run(run())

    assert answer.status == "failed"
    assert answer.answer is None
    assert answer.error == "react response did not prove goal completion"


def test_react_fails_when_the_configured_round_budget_is_exhausted():
    model = SequencedToolModel(
        [
            AIMessage(content="", tool_calls=[{"name": "first", "args": {}, "id": "call-1"}]),
            AIMessage(content="", tool_calls=[{"name": "first", "args": {}, "id": "call-2"}]),
        ]
    )
    runtime = ToolRuntime("observed")

    async def run():
        return await create_execution_mode(
            "react", model=model, tool_runtime=runtime, max_rounds=2
        ).run("Keep going")

    answer = asyncio.run(run())

    assert answer == ExecutionAnswer(None, "failed", error="react round budget exhausted")
    assert len(model.calls) == 2
    assert len(runtime.calls) == 1


def test_react_rejects_a_non_positive_round_budget():
    model = ControlledToolModel(AIMessage(content="{}"))

    try:
        create_execution_mode("react", model=model, tool_runtime=ToolRuntime("unused"), max_rounds=0)
    except ValueError as error:
        assert str(error) == "max_rounds must be a positive integer"
    else:
        raise AssertionError("non-positive React budget should be rejected")


def test_direct_mode_fails_when_the_model_returns_only_whitespace():
    model = ControlledTextModel(["  \n"])

    async def run():
        return await create_execution_mode("direct", model=model).run("Do the work")

    answer = asyncio.run(run())

    assert answer.status == "failed"
    assert answer.answer is None
    assert answer.error == "direct model returned an empty response"
    assert model.calls == [("Do the work", {"tools": None})]


def test_direct_mode_fails_with_a_safe_error_when_model_invocation_raises():
    model = FailingTextModel(RuntimeError("provider secret: do-not-leak"))

    async def run():
        return await create_execution_mode("direct", model=model).run("Do the work")

    answer = asyncio.run(run())

    assert answer.status == "failed"
    assert answer.answer is None
    assert answer.error == "direct model invocation failed"
    assert "do-not-leak" not in answer.error
    assert model.calls == 1


def test_factory_has_one_common_runner_contract_for_each_explicit_mode():
    for mode in ("direct", "tool_agent", "react", "plan_execute"):
        runner = create_execution_mode(mode, model=ControlledTextModel(["unused"])) if mode == "direct" else create_execution_mode(mode)

        assert hasattr(runner, "run")

    try:
        create_execution_mode("router")
    except ValueError as error:
        assert str(error) == "mode must be direct, tool_agent, react, or plan_execute"
    else:
        raise AssertionError("unknown mode should be rejected")
