import asyncio
import json

from langchain_core.messages import AIMessage

import pytest

from agent.runtime_context import DEFAULT_CONTEXT_BUDGET, ContextMaintenanceError, RuntimeContextPolicy


def observation_protocol(payload):
    lines = ["BEGIN OBSERVATION", f"ROUND={payload['round']}"]
    for category in ("confirmed_facts", "reported_errors", "model_inferences"):
        for evidence in payload[category]:
            lines.extend([
                "BEGIN EVIDENCE",
                f"CATEGORY={json.dumps(category)}",
                f"TEXT={json.dumps(evidence['text'])}",
                *[f"TOOL_CALL_ID={json.dumps(call_id)}" for call_id in evidence["tool_call_ids"]],
                "END EVIDENCE",
            ])
    lines.append("END OBSERVATION")
    return "\n".join(lines)


class ObservationModel:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def bind(self, **kwargs):
        self.bind_options = kwargs
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=observation_protocol(self.content) if isinstance(self.content, dict) else self.content)


class BlockingObservationModel(ObservationModel):
    def __init__(self):
        super().__init__(None)
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def ainvoke(self, messages):
        self.calls.append(messages)
        self.started.set()
        await self.release.wait()
        return AIMessage(
            content=observation_protocol(
                {
                    "round": 1,
                    "confirmed_facts": [],
                    "reported_errors": [],
                    "model_inferences": [],
                }
            )
        )


class FirstObservationDelayedModel(BlockingObservationModel):
    def __init__(self):
        super().__init__()
        self.round = 0

    async def ainvoke(self, messages):
        self.calls.append(messages)
        self.round += 1
        response_round = self.round
        if response_round == 1:
            self.started.set()
            await self.release.wait()
        return AIMessage(
            content=observation_protocol(
                {
                    "round": response_round,
                    "confirmed_facts": [
                        {"text": f"fact-{response_round}", "tool_call_ids": [f"call-{response_round}"]}
                    ],
                    "reported_errors": [],
                    "model_inferences": [],
                }
            )
        )


class FailingObservationModel(ObservationModel):
    async def ainvoke(self, messages):
        self.calls.append(messages)
        raise RuntimeError("provider unavailable")


class CancelledObservationModel(ObservationModel):
    def __init__(self):
        super().__init__(None)
        self.started = asyncio.Event()

    async def ainvoke(self, messages):
        self.calls.append(messages)
        self.started.set()
        raise asyncio.CancelledError


def test_observation_lifecycle_accounts_model_use_without_consuming_tool_rounds_or_retrying():
    async def run():
        model = FailingObservationModel({})
        policy = RuntimeContextPolicy(model)

        await policy.record_tool_round(
            1, [{"id": "call-1", "name": "read", "args": {}}], ["done"]
        )
        await asyncio.sleep(0)

        assert policy.model_uses == 1
        assert policy.tool_rounds == 1
        assert len(model.calls) == 1
        assert policy.observation_outcomes[0].status == "provider_error"
        assert "provider unavailable" in policy.observation_outcomes[0].error
        assert policy.assemble({}).raw_tool_results[0].calls[0].result == "done"

    asyncio.run(run())


def test_invalid_observation_is_diagnosable_and_keeps_raw_fallback():
    async def run():
        policy = RuntimeContextPolicy(
            ObservationModel("BEGIN OBSERVATION\nROUND=1\nUNKNOWN=true\nEND OBSERVATION")
        )

        await policy.record_tool_round(
            1, [{"id": "call-1", "name": "read", "args": {}}], ["done"]
        )
        await asyncio.sleep(0)

        outcome = policy.observation_outcomes[0]
        assert outcome.status == "invalid"
        assert "invalid" in outcome.error
        assert policy.model_uses == 1
        assert policy.assemble({}).observations == ()
        assert policy.assemble({}).raw_tool_results[0].calls[0].result == "done"

    asyncio.run(run())


def test_cancelled_observation_is_diagnosable_without_cancelling_the_active_policy():
    async def run():
        model = CancelledObservationModel()
        policy = RuntimeContextPolicy(model)

        await policy.record_tool_round(
            1, [{"id": "call-1", "name": "read", "args": {}}], ["done"]
        )
        await model.started.wait()
        await asyncio.sleep(0)

        assert policy.observation_outcomes[0].status == "cancelled"
        assert policy.model_uses == 1
        assert policy.tool_rounds == 1
        assert policy.assemble({}).raw_tool_results[0].calls[0].result == "done"

    asyncio.run(run())


def test_observation_outcomes_are_forwarded_to_runtime_context_trace():
    class Trace:
        def __init__(self):
            self.outcomes = []

        def runtime_context_observation(self, outcome):
            self.outcomes.append(outcome)

    async def run():
        trace = Trace()
        policy = RuntimeContextPolicy(FailingObservationModel({}), trace=trace)

        await policy.record_tool_round(
            1, [{"id": "call-1", "name": "read", "args": {}}], ["done"]
        )
        await asyncio.sleep(0)

        assert [outcome.status for outcome in trace.outcomes] == [
            "pending",
            "provider_error",
        ]
        assert trace.outcomes[-1].round == 1

    asyncio.run(run())


def test_recording_a_tool_round_exposes_raw_evidence_before_observation_finishes():
    async def run():
        model = BlockingObservationModel()
        policy = RuntimeContextPolicy(model)

        await asyncio.wait_for(
            policy.record_tool_round(
                1,
                [{"id": "call-1", "name": "read", "args": {"path": "README.md"}}],
                ["contents"],
            ),
            timeout=0.1,
        )

        raw = policy.assemble({"goal": "inspect"}).raw_tool_results
        assert [item.round for item in raw] == [1]
        assert raw[0].calls[0].result == "contents"
        assert not model.release.is_set()

        await model.started.wait()
        model.release.set()

    asyncio.run(run())


def test_pending_old_observation_remains_raw_until_a_later_snapshot():
    async def run():
        model = FirstObservationDelayedModel()
        policy = RuntimeContextPolicy(model)

        for round_number in range(1, 5):
            await policy.record_tool_round(
                round_number,
                [{"id": f"call-{round_number}", "name": "read", "args": {}}],
                [f"result-{round_number}"],
            )
        await model.started.wait()

        pending = policy.assemble({"goal": "inspect"})
        assert [item.round for item in pending.raw_tool_results] == [1]
        assert [item.round for item in pending.observations] == [2, 3, 4]

        model.release.set()
        await asyncio.sleep(0.01)

        ready = policy.assemble({"goal": "inspect"})
        assert ready.raw_tool_results == ()
        assert [item.round for item in ready.observations] == [1, 2, 3, 4]
        assert ready.observations[0].source_tool_call_ids == ("call-1",)

    asyncio.run(run())


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
            [{"id": "call-1", "name": "unknown_read", "args": {}}],
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
            content=observation_protocol(
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
        if "Merge the Observations" in messages[0].content:
            payload = json.loads(messages[-1].content)
            observations = payload if isinstance(payload, list) else payload["merge_observations"]
            second_round = observations[1]["round"]
            return AIMessage(
                content=observation_protocol(
                {
                    "round": second_round,
                    "confirmed_facts": [
                        {
                            "text": "merged facts",
                            "tool_call_ids": [
                                call_id
                                for observation in observations
                                for fact in observation["confirmed_facts"]
                                for call_id in fact["tool_call_ids"]
                            ],
                        }
                    ],
                        "reported_errors": [],
                        "model_inferences": [],
                    }
                )
            )
        self.round += 1
        return AIMessage(
            content=observation_protocol(
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
    assert [item.tool_call_id for item in context.raw_tool_results[0].calls] == ["call-2"]
    assert context.raw_tool_results[0].calls[0].result == {"error": "missing"}
    assert context.observations[0].source_tool_call_ids == ("call-1",)
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


def test_observation_request_uses_line_protocol_without_provider_response_format():
    model = ObservationModel(
        {"round": 1, "confirmed_facts": [], "reported_errors": [], "model_inferences": []}
    )
    policy = RuntimeContextPolicy(model)

    async def run():
        await policy.record_tool_round(
            1,
            [{"id": "call-1", "name": "read", "args": {}}],
            ["done"],
        )

    asyncio.run(run())

    assert model.bind_options == {"tools": []}
    contract, evidence = model.calls[0]
    assert "BEGIN OBSERVATION" in contract.content
    assert "Return exactly one JSON object" not in contract.content
    assert json.loads(evidence.content)["calls"][0]["id"] == "call-1"


def test_observation_request_puts_its_fixed_contract_before_settled_raw_evidence():
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

    asyncio.run(
        policy.record_tool_round(
            1,
            [{"id": "call-1", "name": "read", "args": {"path": "VERSION"}}],
            ["1.4.0"],
        )
    )

    contract, evidence = model.calls[0]

    assert "Create the Observation" in contract.content
    assert "call-1" not in contract.content
    assert "VERSION" not in contract.content
    assert "BEGIN OBSERVATION" in contract.content
    assert json.loads(evidence.content) == {
        "round": 1,
        "calls": [
            {
                "id": "call-1",
                "name": "read",
                "args": {"path": "VERSION"},
                "result": "1.4.0",
                "error": None,
            }
        ],
    }
    assert policy.observations[0].confirmed_facts[0].tool_call_ids == ("call-1",)


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

    assert context.raw_tool_results == ()
    assert [item.round for item in context.observations] == [1, 2, 3, 4]
    assert context.observations[0].confirmed_facts[0].text == "fact-1"


def test_policy_uses_raw_fallback_when_observation_schema_is_invalid():
    model = ObservationModel({"round": 1, "confirmed_facts": []})
    policy = RuntimeContextPolicy(model)

    async def run():
        await policy.record_tool_round(1, [{"id": "call-1", "name": "read", "args": {}}], ["done"])
        await asyncio.sleep(0)
        context = policy.assemble({"goal": "inspect"})
        assert context.raw_tool_results[0].calls[0].result == "done"
        assert context.observations == ()

    asyncio.run(run())


def test_policy_uses_raw_fallback_when_observation_evidence_is_invalid():
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
        await asyncio.sleep(0)
        context = policy.assemble({"goal": "inspect"})
        assert context.raw_tool_results[0].calls[0].result == "done"
        assert context.observations == ()

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
    assert context.observations[0].source_round_end == 5
    assert context.observations[0].source_tool_call_ids == (
        "call-1",
        "call-2",
        "call-3",
        "call-4",
        "call-5",
    )
    assert context.raw_tool_results == ()
    merge_contract, merge_evidence = next(
        call for call in model.calls if len(call) == 2 and "Merge the Observations" in call[0].content
    )
    assert "BEGIN OBSERVATION" in merge_contract.content
    assert [item["round"] for item in json.loads(merge_evidence.content)] == [1, 2]


def test_policy_merges_observations_in_round_order_after_out_of_order_completion(monkeypatch):
    monkeypatch.setattr("agent.runtime_context._estimate_tokens", lambda context: 2 if len(context.observations) > 1 else 1)

    class OutOfOrderModel:
        def __init__(self):
            self.release_first = asyncio.Event()
            self.first_started = asyncio.Event()
            self.merge_payloads = []

        def bind(self, **kwargs):
            return self

        async def ainvoke(self, messages):
            payload = json.loads(messages[1].content)
            if isinstance(payload, list):
                self.merge_payloads.append(payload)
                return AIMessage(content=observation_protocol({
                    "round": payload[-1]["round"],
                    "confirmed_facts": [
                        {
                            "text": "merged",
                            "tool_call_ids": [
                                call_id
                                for observation in payload
                                for fact in observation["confirmed_facts"]
                                for call_id in fact["tool_call_ids"]
                            ],
                        }
                    ],
                    "reported_errors": [],
                    "model_inferences": [],
                }))
            if payload["round"] == 1:
                self.first_started.set()
                await self.release_first.wait()
            round_number = payload["round"]
            return AIMessage(content=observation_protocol({
                "round": round_number,
                "confirmed_facts": [{"text": f"fact-{round_number}", "tool_call_ids": [f"call-{round_number}"]}],
                "reported_errors": [],
                "model_inferences": [],
            }))

    async def run():
        model = OutOfOrderModel()
        policy = RuntimeContextPolicy(model, budget=1)
        for round_number in range(1, 6):
            await policy.record_tool_round(
                round_number,
                [{"id": f"call-{round_number}", "name": "read", "args": {}}],
                [f"result-{round_number}"],
            )
        await model.first_started.wait()
        model.release_first.set()
        await asyncio.sleep(0)
        context = await policy.maintain({"goal": "inspect"})
        return model, context

    model, context = asyncio.run(run())

    assert [item["round"] for item in model.merge_payloads[0]] == [1, 2]
    assert context.observations[0].source_round_start == 1
    assert context.observations[0].source_round_end == 5
    assert context.observations[0].source_tool_call_ids == (
        "call-1",
        "call-2",
        "call-3",
        "call-4",
        "call-5",
    )


def test_observation_merge_request_puts_fixed_contract_before_observations(monkeypatch):
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

    asyncio.run(run())

    contract, evidence = next(
        call for call in model.calls if len(call) == 2 and "Merge the Observations" in call[0].content
    )

    assert "Merge the Observations" in contract.content
    assert "call-1" not in contract.content
    assert "call-2" not in contract.content
    assert "BEGIN OBSERVATION" in contract.content
    assert [item["round"] for item in json.loads(evidence.content)] == [1, 2]
    assert [item["confirmed_facts"][0]["tool_call_ids"] for item in json.loads(evidence.content)] == [
        ["call-1"],
        ["call-2"],
    ]


def test_observation_merge_request_omits_provider_response_format(monkeypatch):
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
        await policy.maintain({"goal": "inspect"})

    asyncio.run(run())

    contract, evidence = next(
        call for call in model.calls if len(call) == 2 and "Merge the Observations" in call[0].content
    )

    assert "BEGIN OBSERVATION" in contract.content
    assert "call-1" not in contract.content
    assert [item["round"] for item in json.loads(evidence.content)] == [1, 2]
    assert model.bind_options == {"tools": []}


def test_policy_can_merge_validated_observations_from_recent_rounds(monkeypatch):
    monkeypatch.setattr("agent.runtime_context._estimate_tokens", lambda context: 2)
    policy = RuntimeContextPolicy(MergeObservationModel(), budget=1)

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
