import asyncio
import json

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver

from agent import ExecutionAnswer, Executor, create_execution_mode
from agent.runtime_context import RuntimeContextPolicy
from memory.state import AgentState, Plan, PlanStep


class _ToolRuntime:
    class _Tool:
        name = "record"

    tools = [_Tool()]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def invoke(self, call):
        return f"result-{call['id']}"


class _RoundModel:
    def __init__(self):
        self.calls = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        if self.tools == ():
            return AIMessage(content='BEGIN GOAL_JUDGMENT\nCOMPLETED=true\nEVIDENCE="Five tool results were observed"\nEND GOAL_JUDGMENT')
        if len(self.calls) <= 5:
            round_number = len(self.calls)
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "record",
                        "args": {},
                        "id": f"call-{round_number}",
                    }
                ],
            )
        return AIMessage(content='BEGIN REACT_DECISION\nSTATUS="completed"\nANSWER="The five tool rounds are complete."\nEND REACT_DECISION')


class _FirstObservationBlocked:
    def __init__(self):
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def bind(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            await self.release.wait()
        return AIMessage(
            content=(
                "BEGIN OBSERVATION\nROUND=%d\nAFFECTS_CURRENT_DECISION=false\n"
                "END OBSERVATION"
            )
            % self.calls
        )


def test_react_round_five_uses_raw_fallback_for_pending_round_one():
    async def run():
        context_model = _FirstObservationBlocked()
        model = _RoundModel()
        policy = RuntimeContextPolicy(context_model)
        answer = await create_execution_mode(
            "react",
            model=model,
            tool_runtime=_ToolRuntime(),
            context_policy=policy,
            max_rounds=6,
        ).run("Inspect all five results")
        return answer, model, context_model

    answer, model, context_model = asyncio.run(run())

    assert answer == ExecutionAnswer("The five tool rounds are complete.", "completed")
    assert context_model.started.is_set()
    fifth_context = model.calls[4][-1].content
    assert [f"#### Round {round_number}" in fifth_context for round_number in range(1, 5)] == [
        True,
        True,
        True,
        True,
    ]
    assert "tool_call_id=call-1" in fifth_context
    assert "result-call-1" in fifth_context


class _ExecutorRoundModel:
    def __init__(self):
        self.calls = []
        self.operational_calls = []

    def bind_tools(self, tools, **kwargs):
        return self

    def bind(self, **kwargs):
        return self

    async def ainvoke(self, messages):
        self.calls.append(list(messages))
        system = messages[0].content if messages and hasattr(messages[0], "content") else ""
        if "tool-free completion judge" in system:
            completed = len(self.operational_calls) >= 5
            return AIMessage(content=(
                "BEGIN STEP_COMPLETION_PROGRESS\n"
                f"ALL_COMPLETED={'true' if completed else 'false'}\n"
                + (
                    "BEGIN COMPLETED_CRITERION\nNUMBER=1\nEVIDENCE=\"five results checked\"\n"
                    "END COMPLETED_CRITERION\n"
                    if completed else ""
                )
                + "END STEP_COMPLETION_PROGRESS"
            ))
        if "completed by host-validated evidence" in system:
            return AIMessage(content="The five tool rounds completed.")
        self.operational_calls.append(list(messages))
        if len(self.operational_calls) <= 5:
            round_number = len(self.operational_calls)
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "record",
                        "args": {"value": f"round-{round_number}"},
                        "id": f"call-{round_number}",
                    }
                ],
            )
        return AIMessage(content="The five tool rounds completed.")


class _ExecutorClient:
    class _Session:
        async def __aenter__(self):
            return object()

        async def __aexit__(self, *args):
            return None

    def session(self, name):
        return self._Session()


def _record(value: str) -> str:
    """Return the supplied test value."""
    return value


def test_plan_step_executor_round_five_uses_the_same_raw_fallback_window(monkeypatch):
    async def load_tools(session, **kwargs):
        return [StructuredTool.from_function(_record, name="record")]

    monkeypatch.setattr("agent.executor.MultiServerMCPClient", lambda config: _ExecutorClient())
    monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)

    async def run():
        context_model = _FirstObservationBlocked()
        model = _ExecutorRoundModel()
        saver = InMemorySaver()
        plan = Plan(1, "Inspect results", (PlanStep("inspect", "Inspect", "Results checked"),))
        state = AgentState("Inspect results", plan_history=(plan,))
        async with Executor(
            model=model,
            max_rounds=5,
            context_policy=RuntimeContextPolicy(context_model),
            checkpointer=saver,
            run_id="ticket-05",
        ) as executor:
            outcome = await executor.execute(state, 1, "inspect")
        return outcome, model, context_model, saver

    outcome, model, context_model, saver = asyncio.run(run())

    assert outcome.execution.status == "completed", outcome.execution.error
    assert context_model.started.is_set()
    fifth_context = model.operational_calls[4][1].content
    assert [f"#### Round {round_number}" in fifth_context for round_number in range(1, 5)] == [
        True,
        True,
        True,
        True,
    ]
    assert "tool_call_id=call-1" in fifth_context
    assert "round-1" in fifth_context
    checkpoints = list(
        saver.list({"configurable": {"thread_id": "ticket-05:r1:sinspect"}})
    )
    assert checkpoints
    serialized = repr(checkpoints)
    assert "round-1" not in serialized
    assert "tool_call_id=call-1" not in serialized
