import asyncio
import json

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from agent.execution import ExecutionAnswer, ReactMode
from agent.executor import Executor
from agent.runtime_context import RuntimeContextPolicy
from memory.state import AgentState, Plan, PlanStep


def _observation(round_number, tool_call_id):
    return "\n".join(
        [
            "BEGIN OBSERVATION",
            f"ROUND={round_number}",
            "BEGIN EVIDENCE",
            'CATEGORY="confirmed_facts"',
            'TEXT="write completed"',
            f"TOOL_CALL_ID={json.dumps(tool_call_id)}",
            "END EVIDENCE",
            "END OBSERVATION",
        ]
    )


class BlockingObservationModel:
    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def bind(self, **_kwargs):
        return self

    async def ainvoke(self, messages):
        self.started.set()
        await self.release.wait()
        call_id = json.loads(messages[1].content)["calls"][0]["id"]
        return AIMessage(content=_observation(1, call_id))


class ReactModel:
    def __init__(self):
        self.requests = []
        self.responses = iter(
            (
                AIMessage(
                    content="",
                    tool_calls=[{"name": "write_file", "args": {"path": "note"}, "id": "write-1"}],
                ),
                AIMessage(content="BEGIN NO_TOOL\nEND NO_TOOL"),
                AIMessage(
                    content='BEGIN GOAL_COMPLETION\nANSWER="Done"\nGOAL_SATISFIED=true\nEND GOAL_COMPLETION'
                ),
            )
        )

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages):
        self.requests.append(messages)
        return next(self.responses)


class ReactRuntime:
    tools = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        pass

    async def invoke(self, _call):
        return {"written": "note"}


class Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        pass


class Client:
    def __init__(self, _config):
        pass

    def session(self, _name):
        return Session()


class ExecutorModel:
    def __init__(self):
        self.requests = []
        self.responses = iter(
            (
                AIMessage(
                    content="",
                    tool_calls=[{"name": "write_file", "args": {"path": "note"}, "id": "write-1"}],
                ),
                AIMessage(content="BEGIN NO_TOOL\nEND NO_TOOL"),
                AIMessage(
                    content=(
                        "BEGIN STEP_COMPLETION\nCOMPLETED=true\n"
                        "COMPLETION_CRITERION_MET=true\nRESULT=\"Done\"\nEND STEP_COMPLETION"
                    )
                ),
            )
        )

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages):
        self.requests.append(messages)
        return next(self.responses)


def _step_state():
    return AgentState(
        "Write note",
        plan_history=(Plan(1, "Write note", (PlanStep("write", "Write note", "Note written"),)),),
    )


def test_react_uses_pending_raw_write_evidence_without_waiting_for_observation():
    async def run():
        observation_model = BlockingObservationModel()
        model = ReactModel()

        answer = await ReactMode(
            model,
            ReactRuntime(),
            context_policy=RuntimeContextPolicy(observation_model),
        ).run("Write note")

        assert answer == ExecutionAnswer("Done", "completed")
        assert observation_model.started.is_set()
        assert "tool_call_id=write-1" in model.requests[1][-1].content
        assert '- result: {"written": "note"}' in model.requests[1][-1].content

        observation_model.release.set()
        await asyncio.sleep(0)

    asyncio.run(run())


def test_executor_uses_pending_raw_write_evidence_without_waiting_for_observation(monkeypatch):
    def write_file(path: str) -> str:
        """Write a note."""
        return f"wrote {path}"

    async def load_tools(_session):
        return [StructuredTool.from_function(write_file, name="write_file")]

    async def run():
        observation_model = BlockingObservationModel()
        model = ExecutorModel()
        monkeypatch.setattr("agent.executor.MultiServerMCPClient", Client)
        monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)

        async with Executor(
            model=model,
            context_policy=RuntimeContextPolicy(observation_model),
        ) as executor:
            outcome = await executor.execute(_step_state(), 1, "write")

        assert outcome.execution.status == "completed"
        assert observation_model.started.is_set()
        assert "tool_call_id=write-1" in model.requests[1][-1].content
        assert '- result: "wrote note"' in model.requests[1][-1].content

        observation_model.release.set()
        await asyncio.sleep(0)

    asyncio.run(run())
