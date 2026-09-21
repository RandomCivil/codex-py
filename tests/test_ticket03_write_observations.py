import asyncio
import json

from langchain_core.messages import AIMessage

from agent.runtime_context import RuntimeContextPolicy


def _observation(round_number, tool_call_id, text):
    return "\n".join(
        [
            "BEGIN OBSERVATION",
            f"ROUND={round_number}",
            "BEGIN EVIDENCE",
            'CATEGORY="confirmed_facts"',
            f"TEXT={json.dumps(text)}",
            f"TOOL_CALL_ID={json.dumps(tool_call_id)}",
            "END EVIDENCE",
            "END OBSERVATION",
        ]
    )


class ControlledObservationModel:
    def __init__(self):
        self.calls = []
        self.started = {}
        self.release = {}

    def bind(self, **kwargs):
        self.bind_options = kwargs
        return self

    async def ainvoke(self, messages):
        payload = json.loads(messages[1].content)
        call = payload["calls"][0]
        call_id = call["id"]
        self.calls.append(messages)
        self.started.setdefault(call_id, asyncio.Event()).set()
        await self.release.setdefault(call_id, asyncio.Event()).wait()
        return AIMessage(content=_observation(payload["round"], call_id, f"observed {call_id}"))


class OutcomeObservationModel:
    def __init__(self):
        self.calls = []

    def bind(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        payload = json.loads(messages[1].content)
        call_id = payload["calls"][0]["id"]
        self.calls.append(call_id)
        if call_id == "provider-error":
            raise RuntimeError("provider unavailable")
        if call_id == "cancelled":
            raise asyncio.CancelledError
        if call_id == "invalid-range":
            return AIMessage(
                content="\n".join(
                    (
                        "BEGIN OBSERVATION",
                        f"ROUND={payload['round']}",
                        "SOURCE_ROUND_START=999",
                        "SOURCE_ROUND_END=1000",
                        "BEGIN EVIDENCE",
                        'CATEGORY="confirmed_facts"',
                        'TEXT="untrusted"',
                        f"TOOL_CALL_ID={json.dumps(call_id)}",
                        "END EVIDENCE",
                        "END OBSERVATION",
                    )
                )
            )
        return AIMessage(content=_observation(payload["round"], "wrong-source", "untrusted"))


def test_each_write_call_has_an_independent_nonblocking_observation_and_fallback():
    async def run():
        model = ControlledObservationModel()
        policy = RuntimeContextPolicy(model)

        await policy.record_tool_round(
            1,
            [
                {"id": "read-1", "name": "read_file", "args": {"path": "a"}},
                {"id": "patch-1", "name": "apply_patch", "args": {"patch": "a"}},
                {"id": "write-1", "name": "write_file", "args": {"path": "b"}},
            ],
            ["read result", "patch result", "write result"],
        )

        await model.started["patch-1"].wait()
        await model.started["write-1"].wait()
        pending = policy.assemble({})
        assert [call.tool_call_id for result in pending.raw_tool_results for call in result.calls] == [
            "read-1",
            "patch-1",
            "write-1",
        ]
        assert len(model.calls) == 2
        assert {outcome.tool_call_id for outcome in policy.observation_outcomes} == {
            "patch-1",
            "write-1",
        }

        model.release["patch-1"].set()
        await asyncio.sleep(0.01)
        ready = policy.assemble({})
        assert [call.tool_call_id for result in ready.raw_tool_results for call in result.calls] == [
            "read-1",
            "write-1",
        ]
        assert [observation.source_tool_call_ids for observation in ready.observations] == [("patch-1",)]

        model.release["write-1"].set()
        await asyncio.sleep(0.01)

    asyncio.run(run())


def test_failed_cancelled_and_invalid_observations_keep_per_call_raw_fallback_without_retry():
    async def run():
        model = OutcomeObservationModel()
        policy = RuntimeContextPolicy(model)

        await policy.record_tool_round(
            4,
            [
                {"id": "provider-error", "name": "write_file", "args": {}},
                {"id": "cancelled", "name": "apply_patch", "args": {}},
                {"id": "invalid", "name": "exec", "args": {"command": "touch marker"}},
                {"id": "invalid-range", "name": "write_file", "args": {}},
            ],
            ["provider result", "cancel result", "invalid result", "invalid range result"],
        )
        await asyncio.sleep(0.01)

        assert model.calls == ["provider-error", "cancelled", "invalid", "invalid-range"]
        assert {(outcome.tool_call_id, outcome.status) for outcome in policy.observation_outcomes} == {
            ("provider-error", "provider_error"),
            ("cancelled", "cancelled"),
            ("invalid", "invalid"),
            ("invalid-range", "invalid"),
        }
        context = policy.assemble({})
        assert [
            call.tool_call_id
            for result in context.raw_tool_results
            for call in result.calls
        ] == ["provider-error", "cancelled", "invalid", "invalid-range"]
        assert context.observations == ()

    asyncio.run(run())


def test_completed_observations_keep_the_batch_request_order_when_completion_is_out_of_order():
    async def run():
        model = ControlledObservationModel()
        policy = RuntimeContextPolicy(model)
        await policy.record_tool_round(
            2,
            [
                {"id": "first", "name": "write_file", "args": {}},
                {"id": "second", "name": "apply_patch", "args": {}},
            ],
            ["first result", "second result"],
        )
        await model.started["first"].wait()
        await model.started["second"].wait()

        model.release["second"].set()
        await asyncio.sleep(0.01)
        model.release["first"].set()
        await asyncio.sleep(0.01)

        assert [
            observation.source_tool_call_ids[0]
            for observation in policy.assemble({}).observations
        ] == ["first", "second"]

    asyncio.run(run())
