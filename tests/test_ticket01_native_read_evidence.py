import asyncio

from agent.runtime_context import RuntimeContextPolicy


class RecordingModel:
    def __init__(self):
        self.calls = []

    def bind(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        self.calls.append(messages)
        raise AssertionError("native read evidence must not request an Observation")


def test_native_tool_calls_keep_their_own_evidence_lifecycle():
    async def run():
        model = RecordingModel()
        policy = RuntimeContextPolicy(model)

        await policy.record_tool_round(
            1,
            [
                {"id": "list-1", "name": "list_dir", "args": {"path": "."}},
                {"id": "grep-1", "name": "grep", "args": {"pattern": "policy"}},
                {"id": "file-1", "name": "read_file", "args": {"path": "README.md"}},
            ],
            ["old listing", "old match", "old file"],
        )
        for round_number in range(2, 5):
            await policy.record_tool_round(
                round_number,
                [{"id": f"glob-{round_number}", "name": "glob", "args": {}}],
                [f"listing-{round_number}"],
            )

        context = policy.assemble({"goal": "inspect"})
        visible_ids = [
            call.tool_call_id
            for result in context.raw_tool_results
            for call in result.calls
        ]

        assert visible_ids == ["grep-1", "file-1", "glob-4"]
        retained = {
            call.tool_call_id: call
            for result in context.raw_tool_results
            for call in result.calls
        }
        assert retained["grep-1"].arguments == {"pattern": "policy"}
        assert retained["grep-1"].result == "old match"
        assert retained["file-1"].arguments == {"path": "README.md"}
        assert retained["file-1"].result == "old file"
        assert model.calls == []

    asyncio.run(run())
