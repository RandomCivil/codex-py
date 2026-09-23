import asyncio
import json

import pytest
from langchain_core.messages import AIMessage

from agent.execution import ExecutionAnswer, ReactMode
from agent.runtime_context import RuntimeContextPolicy


def _observation(
    round_number,
    tool_call_id,
    *,
    impact=True,
    targets=("confirmed_facts",),
    include_evidence=True,
):
    lines = [
        "BEGIN OBSERVATION",
        f"ROUND={round_number}",
        f"AFFECTS_CURRENT_DECISION={str(impact).lower()}",
    ]
    lines.extend(f"AFFECTED_TARGETS={json.dumps(target)}" for target in targets)
    if include_evidence:
        lines.extend(
            (
                "BEGIN EVIDENCE",
                'CATEGORY="confirmed_facts"',
                'TEXT="the note was written"',
                f"TOOL_CALL_ID={json.dumps(tool_call_id)}",
                "END EVIDENCE",
            )
        )
    lines.append("END OBSERVATION")
    return "\n".join(lines)


class _ContextModel:
    def __init__(self, *, impact=True, targets=("confirmed_facts",), include_evidence=True):
        self.requests = []
        self.impact = impact
        self.targets = targets
        self.include_evidence = include_evidence

    def bind(self, **_kwargs):
        return self

    async def ainvoke(self, messages):
        self.requests.append(messages)
        payload = json.loads(messages[1].content)
        call = payload["calls"][0]
        response = AIMessage(
            content=_observation(
                payload["round"],
                call["id"],
                impact=self.impact,
                targets=self.targets,
                include_evidence=self.include_evidence,
            )
        )
        return response


class _ReactModel:
    def __init__(self, calls=None):
        self.requests = []
        self.responses = iter(
            (
                AIMessage(
                    content="",
                    tool_calls=calls or [
                        {
                            "name": "write_file",
                            "args": {"path": "note.md", "content": "hello"},
                            "id": "write-1",
                        }
                    ],
                ),
                AIMessage(content="Done"),
            )
        )

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages):
        self.requests.append(list(messages))
        return next(self.responses)


class _Runtime:
    tools = []

    def __init__(self, result=None):
        self.result = result or {"written": "note.md"}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def invoke(self, _call):
        return self.result


def test_react_observation_uses_decision_summary_and_replaces_raw_with_positive_impact():
    async def run():
        context_model = _ContextModel()
        model = _ReactModel()
        answer = await ReactMode(
            model,
            _Runtime(),
            context_policy=RuntimeContextPolicy(context_model),
        ).run("Write and verify the note")
        return answer, context_model, model

    answer, context_model, model = asyncio.run(run())

    assert answer == ExecutionAnswer("Done", "completed")
    observation_payload = json.loads(context_model.requests[0][1].content)
    assert observation_payload["decision_summary"] == {
        "goal": "Write and verify the note",
        "execution_mode": "react",
        "plan_step": None,
        "completion_criterion": None,
        "tool": {"name": "write_file", "arguments": {"path": "note.md", "content": "hello"}},
    }
    assert context_model.requests[0][0].content.startswith("Return exactly one UTF-8 Line Protocol block")
    assert "AFFECTS_CURRENT_DECISION" in context_model.requests[0][0].content
    rendered = model.requests[1][-1].content
    assert "the note was written" in rendered
    assert "write-1" in rendered
    assert '"written": "note.md"' not in rendered


def test_positive_observation_replaces_raw_fallback_after_it_finishes():
    async def run():
        context_model = _ContextModel()
        policy = RuntimeContextPolicy(context_model)
        await policy.record_tool_round(
            1,
            [{"name": "write_file", "args": {"path": "note.md"}, "id": "write-1"}],
            [{"written": "note.md"}],
            decision_context={
                "goal": "Write and verify the note",
                "execution_mode": "react",
                "plan_step": None,
                "completion_criterion": None,
            },
        )
        await asyncio.sleep(0.01)
        context = policy.assemble({"goal": "Write and verify the note"})
        return context, policy

    context, policy = asyncio.run(run())

    assert policy.observation_outcomes[0].status == "succeeded", policy.observation_outcomes
    assert context.observations[0].affects_current_decision is True
    assert context.observations[0].affected_targets == ("confirmed_facts",)
    assert context.observations[0].source_tool_call_ids == ("write-1",)
    assert context.raw_tool_results == ()


def test_negative_impact_keeps_only_the_existing_raw_correction_window():
    async def run():
        context_model = _ContextModel(impact=False, targets=(), include_evidence=False)
        policy = RuntimeContextPolicy(context_model, raw_rounds=2)
        await policy.record_tool_round(
            1,
            [{"name": "write_file", "args": {"path": "irrelevant"}, "id": "negative-1"}],
            ["unrelated result"],
            decision_context={"goal": "Inspect target", "execution_mode": "react"},
        )
        await asyncio.sleep(0.01)
        inside_window = policy.assemble({})
        await policy.record_tool_round(
            2,
            [{"name": "list_dir", "args": {}, "id": "listing-2"}],
            ["directory listing"],
        )
        await policy.record_tool_round(
            3,
            [{"name": "list_dir", "args": {}, "id": "listing-3"}],
            ["directory listing"],
        )
        outside_window = policy.assemble({})
        return inside_window, outside_window, policy

    inside_window, outside_window, policy = asyncio.run(run())

    assert inside_window.observations == ()
    assert "unrelated result" in repr(inside_window.raw_tool_results)
    assert outside_window.observations == ()
    assert "negative-1" not in repr(outside_window.raw_tool_results)
    assert "negative-1" in repr(policy.raw_tool_results)


def test_react_sends_one_decision_summary_for_each_successful_observation_call_including_exec():
    async def run():
        calls = [
            {"name": "write_file", "args": {"path": "a"}, "id": "write-1"},
            {"name": "apply_patch", "args": {"patch": "b"}, "id": "patch-1"},
            {"name": "exec", "args": {"command": "touch c"}, "id": "exec-1"},
        ]
        context_model = _ContextModel()
        model = _ReactModel(calls)
        answer = await ReactMode(
            model,
            _Runtime(),
            context_policy=RuntimeContextPolicy(context_model),
        ).run("Update three files")
        await asyncio.sleep(0.01)
        return answer, context_model

    answer, context_model = asyncio.run(run())

    assert answer == ExecutionAnswer("Done", "completed")
    summaries = {
        payload["calls"][0]["id"]: payload["decision_summary"]
        for payload in (json.loads(request[1].content) for request in context_model.requests)
    }
    assert summaries == {
        "write-1": {
                "goal": "Update three files",
                "execution_mode": "react",
                "plan_step": None,
                "completion_criterion": None,
                "tool": {"name": "write_file", "arguments": {"path": "a"}},
            },
        "patch-1": {
                "goal": "Update three files",
                "execution_mode": "react",
                "plan_step": None,
                "completion_criterion": None,
                "tool": {"name": "apply_patch", "arguments": {"patch": "b"}},
            },
        "exec-1": {
                "goal": "Update three files",
                "execution_mode": "react",
                "plan_step": None,
                "completion_criterion": None,
                "tool": {"name": "exec", "arguments": {"command": "touch c"}},
            },
    }


@pytest.mark.parametrize(
    "response",
    [
        "BEGIN OBSERVATION\nROUND=1\nAFFECTS_CURRENT_DECISION=true\nEND OBSERVATION",
        "BEGIN OBSERVATION\nROUND=1\nAFFECTS_CURRENT_DECISION=true\n"
        'AFFECTED_TARGETS="confirmed_facts"\nEND OBSERVATION',
        "BEGIN OBSERVATION\nROUND=1\nAFFECTS_CURRENT_DECISION=false\n"
        'AFFECTED_TARGETS="summary"\nEND OBSERVATION',
        "BEGIN OBSERVATION\nROUND=1\nAFFECTS_CURRENT_DECISION=true\n"
        'AFFECTED_TARGETS="unsupported"\nEND OBSERVATION',
    ],
)
def test_invalid_decision_impact_targets_keep_raw_fallback(response):
    class InvalidModel:
        def bind(self, **_kwargs):
            return self

        async def ainvoke(self, _messages):
            return AIMessage(content=response)

    async def run():
        policy = RuntimeContextPolicy(InvalidModel())
        await policy.record_tool_round(
            1,
            [{"name": "write_file", "args": {}, "id": "invalid-1"}],
            ["write result"],
        )
        await asyncio.sleep(0.01)
        return policy

    policy = asyncio.run(run())

    assert policy.observation_outcomes[0].status == "invalid"
    assert policy.observations == ()
    assert policy.assemble({}).raw_tool_results[0].calls[0].tool_call_id == "invalid-1"


def test_merging_observations_keeps_positive_impact_and_unions_affected_targets():
    class MergeModel(_ContextModel):
        async def ainvoke(self, messages):
            self.requests.append(messages)
            payload = json.loads(messages[1].content)
            if isinstance(payload, list):
                call_ids = [
                    call_id
                    for observation in payload
                    for evidence in observation["confirmed_facts"]
                    for call_id in evidence["tool_call_ids"]
                ]
                return AIMessage(
                    content=_observation(
                        payload[-1]["round"],
                        call_ids[0],
                        targets=("confirmed_facts",),
                    ).replace(
                        f'TOOL_CALL_ID="{call_ids[0]}"',
                        "\n".join(f'TOOL_CALL_ID="{call_id}"' for call_id in call_ids),
                    )
                )
            call = payload["calls"][0]
            target = "confirmed_facts" if call["id"] == "merge-1" else "summary"
            return AIMessage(
                content=_observation(payload["round"], call["id"], targets=(target,))
            )

    async def run():
        model = MergeModel()
        policy = RuntimeContextPolicy(model, budget=1)
        # Force a budget merge after both positive Observations are ready.
        import agent.runtime_context as runtime_context

        original_estimator = runtime_context._estimate_tokens
        runtime_context._estimate_tokens = lambda context: (
            2 if len(context.observations) > 1 else 1
        )
        try:
            for number in range(1, 3):
                call_id = f"merge-{number}"
                await policy.record_tool_round(
                    number,
                    [{"name": "write_file", "args": {}, "id": call_id}],
                    [f"result-{number}"],
                )
            await asyncio.sleep(0.01)
            return await policy.maintain({})
        finally:
            runtime_context._estimate_tokens = original_estimator

    context = asyncio.run(run())

    assert len(context.observations) == 1
    assert context.observations[0].affects_current_decision is True
    assert context.observations[0].affected_targets == ("confirmed_facts", "summary")
