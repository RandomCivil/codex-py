import asyncio
from types import SimpleNamespace

from agent.configuration import ComponentProviderConfiguration, ProviderConfiguration
from agent.execution import ExecutionAnswer, create_execution_mode
from llm.llm import LLM


class EmptyRuntime:
    tools = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class ConfiguredResponsesModel(LLM):
    created = []

    def __init__(self, *args, **kwargs):
        self._model_name = args[2]
        self.requests = []
        type(self).created.append(self)

    async def stream_events(self, input, *, instructions=None, tools=None):
        self.requests.append({"input": input, "instructions": instructions, "tools": tools})
        prompt = repr(input)
        if "Completion judge" in prompt or "GOAL_JUDGMENT" in prompt:
            output_text = 'BEGIN GOAL_JUDGMENT\nCOMPLETED=true\nEVIDENCE="Done"\nEND GOAL_JUDGMENT'
        elif "REACT_DECISION" in prompt:
            output_text = 'BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="Done"\nEND REACT_DECISION'
        else:
            output_text = 'BEGIN ANSWER\nTEXT="Done"\nEND ANSWER'
        yield SimpleNamespace(
            type="response.completed",
            response=SimpleNamespace(
                output_text=output_text,
                output=[],
                usage={"input_tokens": 3, "output_tokens": 2},
            ),
        )


def test_configured_tool_modes_use_shared_responses_path_and_keep_results(monkeypatch):
    ConfiguredResponsesModel.created.clear()
    monkeypatch.setattr("agent.execution.LLM", ConfiguredResponsesModel)
    provider = ProviderConfiguration("https://provider.test/v1", "key", "model")
    configuration = ComponentProviderConfiguration(
        planner=provider,
        executor=provider,
        task_analyzer=provider,
        tool_agent=provider,
        react=provider,
    )

    tool_agent = create_execution_mode("tool_agent", configuration=configuration, tool_runtime=EmptyRuntime())
    react = create_execution_mode("react", configuration=configuration, tool_runtime=EmptyRuntime())

    assert isinstance(tool_agent._model, LLM)
    assert isinstance(react._model, LLM)
    assert asyncio.run(tool_agent.run("Answer")) == ExecutionAnswer("Done", "completed")
    assert asyncio.run(react.run("Finish")) == ExecutionAnswer("Done", "completed")
    assert len(ConfiguredResponsesModel.created) == 2
    assert all(model.requests for model in ConfiguredResponsesModel.created)


def test_configured_react_leaves_request_attribution_to_each_call_site():
    provider = ProviderConfiguration("https://provider.test/v1", "key", "model")
    configuration = ComponentProviderConfiguration(
        planner=provider,
        executor=provider,
        task_analyzer=provider,
        react=provider,
    )

    react = create_execution_mode("react", configuration=configuration, tool_runtime=EmptyRuntime())

    assert react._model._on_request is None
