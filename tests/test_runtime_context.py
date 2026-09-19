import asyncio
import json

from langchain_core.messages import AIMessage

import pytest

from agent.runtime_context import DEFAULT_CONTEXT_BUDGET, ContextMaintenanceError, RuntimeContextPolicy


class ObservationModel:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def bind(self, **kwargs):
        self.bind_options = kwargs
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=json.dumps(self.content))


def test_policy_logs_observation_model_response():
    class Trace:
        def __init__(self):
            self.responses = []

        def llm_response(self, response):
            self.responses.append(response)

    trace = Trace()
    model = ObservationModel(
        {
            "round": 1,
            "confirmed_facts": [],
            "reported_errors": [],
            "model_inferences": [],
        }
    )
    policy = RuntimeContextPolicy(model, trace=trace)

    asyncio.run(
        policy.record_tool_round(
            1,
            [{"id": "call-1", "name": "read_file", "args": {}}],
            ["ok"],
        )
    )

    assert len(trace.responses) == 1


class SequencedObservationModel(ObservationModel):
    def __init__(self):
        super().__init__(None)
        self.round = 0

    async def ainvoke(self, messages):
        self.round += 1
        self.calls.append(messages)
        return AIMessage(
            content=json.dumps(
                {
                    "round": self.round,
                    "confirmed_facts": [
                        {"text": f"fact-{self.round}", "tool_call_ids": [f"call-{self.round}"]}
                    ],
                    "reported_errors": [],
                    "model_inferences": [],
                }
            )
        )


class MergeObservationModel(SequencedObservationModel):
    async def ainvoke(self, messages):
        self.calls.append(messages)
        if "merge_observations" in messages[0].content:
            payload = json.loads(messages[0].content)
            second_round = payload["merge_observations"][1]["round"]
            return AIMessage(
                content=json.dumps(
                    {
                        "round": second_round,
                        "confirmed_facts": [
                            {"text": "facts 1-2", "tool_call_ids": ["call-1", "call-2"]}
                        ],
                        "reported_errors": [],
                        "model_inferences": [],
                    }
                )
            )
        self.round += 1
        return AIMessage(
            content=json.dumps(
                {
                    "round": self.round,
                    "confirmed_facts": [],
                    "reported_errors": [],
                    "model_inferences": [],
                }
            )
        )


def test_policy_records_complete_round_and_validated_observation():
    model = ObservationModel(
        {
            "round": 1,
            "confirmed_facts": [
                {"text": "version is 1.4.0", "tool_call_ids": ["call-1"]}
            ],
            "reported_errors": [],
            "model_inferences": [],
        }
    )
    policy = RuntimeContextPolicy(model)

    async def run():
        await policy.record_tool_round(
            1,
            [
                {"id": "call-1", "name": "read", "args": {"path": "pyproject.toml"}},
                {"id": "call-2", "name": "stat", "args": {"path": "README.md"}},
            ],
            ["version is 1.4.0", {"error": "missing"}],
        )

    asyncio.run(run())

    context = policy.assemble(durable_state={"goal": "inspect"})
    assert context.raw_tool_results[0].round == 1
    assert [item.tool_call_id for item in context.raw_tool_results[0].calls] == ["call-1", "call-2"]
    assert context.raw_tool_results[0].calls[0].arguments == {"path": "pyproject.toml"}
    assert context.raw_tool_results[0].calls[1].result == {"error": "missing"}
    assert context.observations == ()
    assert policy.observations[0].round == 1
    assert policy.observations[0].confirmed_facts[0].tool_call_ids == ("call-1",)
    assert model.bind_options["tools"] == []


def test_runtime_context_renders_as_prompt_sections():
    policy = RuntimeContextPolicy(ObservationModel({}))
    context = policy.assemble(
        durable_state={"goal": "inspect"}
    )

    prompt = context.as_messages()[0].content

    assert "### Durable state\n{\"goal\": \"inspect\"}" in prompt
    assert "### Observations\n- (none)" in prompt
    assert "### Raw tool results\n- (none)" in prompt


def test_policy_retains_explicit_tool_failures_with_the_raw_result():
    model = ObservationModel(
        {"round": 1, "confirmed_facts": [], "reported_errors": [], "model_inferences": []}
    )
    policy = RuntimeContextPolicy(model)

    async def run():
        await policy.record_tool_round(
            1,
            [{"id": "call-1", "name": "exec", "args": {"command": "false"}}],
            [{"returncode": 1, "stderr": "failed"}],
            ["command exited with status 1"],
        )

    asyncio.run(run())

    call = policy.assemble({"goal": "inspect"}).raw_tool_results[0].calls[0]
    assert call.result == {"returncode": 1, "stderr": "failed"}
    assert call.error == "command exited with status 1"
    assert "reported_errors" in model.calls[0][0].content


def test_observation_request_uses_the_component_response_format():
    model = ObservationModel(
        {"round": 1, "confirmed_facts": [], "reported_errors": [], "model_inferences": []}
    )
    policy = RuntimeContextPolicy(model, response_format="json_object")

    async def run():
        await policy.record_tool_round(
            1,
            [{"id": "call-1", "name": "read", "args": {}}],
            ["done"],
        )

    asyncio.run(run())

    assert model.bind_options == {"tools": [], "response_format": {"type": "json_object"}}
    prompt = json.loads(model.calls[0][0].content)["instruction"]
    assert "Return exactly one JSON object" in prompt
    assert '"round": 7' in prompt
    assert '"tool_call_ids": ["call-7"]' in prompt


def test_policy_moves_only_the_fourth_oldest_round_to_observation_context():
    model = SequencedObservationModel()
    policy = RuntimeContextPolicy(model)

    async def run():
        for round_number in range(1, 5):
            await policy.record_tool_round(
                round_number,
                [{"id": f"call-{round_number}", "name": "read", "args": {"round": round_number}}],
                [f"result-{round_number}"],
            )

    asyncio.run(run())
    context = policy.assemble(durable_state={"goal": "inspect"})

    assert [item.round for item in context.raw_tool_results] == [2, 3, 4]
    assert [item.round for item in context.observations] == [1]
    assert context.observations[0].confirmed_facts[0].text == "fact-1"


def test_policy_fails_when_observation_schema_is_invalid():
    model = ObservationModel({"round": 1, "confirmed_facts": []})
    policy = RuntimeContextPolicy(model)

    async def run():
        await policy.record_tool_round(1, [{"id": "call-1", "name": "read", "args": {}}], ["done"])

    with pytest.raises(ContextMaintenanceError, match="Observation schema"):
        asyncio.run(run())


def test_policy_rejects_invalid_observation_evidence_types():
    model = ObservationModel(
        {
            "round": 1,
            "confirmed_facts": [{"text": 7, "tool_call_ids": "call-1"}],
            "reported_errors": [],
            "model_inferences": [],
        }
    )
    policy = RuntimeContextPolicy(model)

    async def run():
        await policy.record_tool_round(1, [{"id": "call-1", "name": "read", "args": {}}], ["done"])

    with pytest.raises(ContextMaintenanceError, match="evidence"):
        asyncio.run(run())


def test_policy_uses_default_budget_and_accepts_component_override():
    model = ObservationModel(
        {"round": 1, "confirmed_facts": [], "reported_errors": [], "model_inferences": []}
    )

    assert RuntimeContextPolicy(model).budget == DEFAULT_CONTEXT_BUDGET
    assert RuntimeContextPolicy(model, budget=4096).budget == 4096


def test_policy_merges_oldest_expired_observations_and_preserves_source_range(monkeypatch):
    monkeypatch.setattr("agent.runtime_context._estimate_tokens", lambda context: 2 if len(context.observations) > 1 else 1)
    model = MergeObservationModel()
    policy = RuntimeContextPolicy(model, budget=1)

    async def run():
        for round_number in range(1, 6):
            await policy.record_tool_round(
                round_number,
                [{"id": f"call-{round_number}", "name": "read", "args": {}}],
                [f"result-{round_number}"],
            )
        return await policy.maintain({"goal": "inspect"})

    context = asyncio.run(run())

    assert context.observations[0].source_round_start == 1
    assert context.observations[0].source_round_end == 2
    assert context.observations[0].source_tool_call_ids == ("call-1", "call-2")
    assert [item.round for item in context.raw_tool_results] == [3, 4, 5]
    merge_request = next(
        json.loads(call[0].content) for call in model.calls if "merge_observations" in call[0].content
    )
    assert "Return exactly one JSON object" in merge_request["instruction"]


def test_policy_fails_instead_of_merging_an_observation_with_a_recent_raw_round(monkeypatch):
    monkeypatch.setattr("agent.runtime_context._estimate_tokens", lambda context: 2)
    policy = RuntimeContextPolicy(SequencedObservationModel(), budget=1)

    async def run():
        for round_number in range(1, 5):
            await policy.record_tool_round(
                round_number,
                [{"id": f"call-{round_number}", "name": "read", "args": {}}],
                [f"result-{round_number}"],
            )
        await policy.maintain({"goal": "inspect"})

    with pytest.raises(ContextMaintenanceError, match="exceeds"):
        asyncio.run(run())
