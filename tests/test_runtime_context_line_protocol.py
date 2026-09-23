import asyncio
import json

from langchain_core.messages import AIMessage

from agent.runtime_context import RuntimeContextPolicy


class ObservationLineProtocolModel:
    def __init__(self, response: str):
        self.response = response
        self.calls = []
        self.bind_options = None

    def bind(self, **kwargs):
        self.bind_options = kwargs
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.response)


class MergeLineProtocolModel(ObservationLineProtocolModel):
    def __init__(self):
        super().__init__("" )
        self.round = 0

    async def ainvoke(self, messages):
        self.calls.append(messages)
        payload = json.loads(messages[1].content)
        if isinstance(payload, list):
            source_ids = [
                call_id
                for observation in payload
                for evidence in observation["confirmed_facts"]
                for call_id in evidence["tool_call_ids"]
            ]
            return AIMessage(
                content="\n".join(
                    [
                        "BEGIN OBSERVATION",
                        f"ROUND={payload[-1]['round']}",
                        f"SOURCE_ROUND_START={payload[0]['source_round_start']}",
                        f"SOURCE_ROUND_END={payload[-1]['source_round_end']}",
                        "AFFECTS_CURRENT_DECISION=true",
                        'AFFECTED_TARGETS="confirmed_facts"',
                        "BEGIN EVIDENCE",
                        'CATEGORY="confirmed_facts"',
                        'TEXT="merged facts"',
                        *[f'TOOL_CALL_ID="{call_id}"' for call_id in source_ids],
                        "END EVIDENCE",
                        "END OBSERVATION",
                    ]
                )
            )
        self.round += 1
        return AIMessage(
            content="\n".join(
                [
                    "BEGIN OBSERVATION",
                    f"ROUND={self.round}",
                    "AFFECTS_CURRENT_DECISION=true",
                    'AFFECTED_TARGETS="confirmed_facts"',
                    "BEGIN EVIDENCE",
                    'CATEGORY="confirmed_facts"',
                    f'TEXT="fact-{self.round}"',
                    f'TOOL_CALL_ID="call-{self.round}"',
                    "END EVIDENCE",
                    "END OBSERVATION",
                ]
            )
        )


def test_valid_observation_line_protocol_is_decoded_without_provider_response_format():
    model = ObservationLineProtocolModel(
        "\n".join(
            [
                "BEGIN OBSERVATION",
                "ROUND=1",
                "AFFECTS_CURRENT_DECISION=true",
                'AFFECTED_TARGETS="confirmed_facts"',
                "BEGIN EVIDENCE",
                'CATEGORY="confirmed_facts"',
                'TEXT="version is 1.4.0"',
                'TOOL_CALL_ID="call-1"',
                "END EVIDENCE",
                "END OBSERVATION",
            ]
        )
    )
    policy = RuntimeContextPolicy(model)

    async def run():
        await policy.record_tool_round(
            1, [{"id": "call-1", "name": "read", "args": {}}], ["1.4.0"]
        )

    asyncio.run(run())

    assert policy.observations[0].round == 1
    assert policy.observations[0].confirmed_facts[0].text == "version is 1.4.0"
    assert policy.observations[0].source_tool_call_ids == ("call-1",)
    assert model.bind_options == {"tools": []}
    assert "BEGIN OBSERVATION" in model.calls[0][0].content


def test_malformed_observation_line_protocol_keeps_raw_fallback():
    model = ObservationLineProtocolModel(
        "BEGIN OBSERVATION\nROUND=1\nAFFECTS_CURRENT_DECISION=true\n"
        'AFFECTED_TARGETS="confirmed_facts"\nBEGIN EVIDENCE\nTEXT=7\nEND EVIDENCE\nEND OBSERVATION'
    )
    policy = RuntimeContextPolicy(model)

    async def run():
        await policy.record_tool_round(
            1, [{"id": "call-1", "name": "read", "args": {}}], ["done"]
        )
        await asyncio.sleep(0)

    asyncio.run(run())

    assert policy.observation_outcomes[0].status == "invalid"
    assert policy.observations == ()
    assert policy.assemble({}).raw_tool_results[0].calls[0].result == "done"


def test_observation_evidence_requires_tool_call_id():
    model = ObservationLineProtocolModel(
        "BEGIN OBSERVATION\nROUND=1\nAFFECTS_CURRENT_DECISION=true\n"
        'AFFECTED_TARGETS="confirmed_facts"\n'
        "BEGIN EVIDENCE\nCATEGORY=\"confirmed_facts\"\nTEXT=\"fact\"\n"
        "END EVIDENCE\nEND OBSERVATION"
    )
    policy = RuntimeContextPolicy(model)

    async def run():
        await policy.record_tool_round(
            1, [{"id": "call-1", "name": "read", "args": {}}], ["done"]
        )
        await asyncio.sleep(0)

    asyncio.run(run())

    assert policy.observation_outcomes[0].status == "invalid"
    assert policy.observation_outcomes[0].error == (
        "Observation Line Protocol is invalid: evidence"
    )


def test_observation_merge_decodes_source_range_and_preserves_tool_call_provenance(monkeypatch):
    monkeypatch.setattr(
        "agent.runtime_context._estimate_tokens",
        lambda context: 2 if len(context.observations) > 1 else 1,
    )
    model = MergeLineProtocolModel()
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

    merged = context.observations[0]
    assert merged.source_round_start == 1
    assert merged.source_round_end == 5
    assert merged.source_tool_call_ids == (
        "call-1",
        "call-2",
        "call-3",
        "call-4",
        "call-5",
    )
    assert "BEGIN OBSERVATION" in next(
        call[0].content for call in model.calls if "Merge the Observations" in call[0].content
    )
