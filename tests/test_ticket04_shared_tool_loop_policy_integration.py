import asyncio
import json

from langchain_core.messages import AIMessage, ToolMessage
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
            "AFFECTS_CURRENT_DECISION=true",
            'AFFECTED_TARGETS="confirmed_facts"',
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
                AIMessage(content="Done"),
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


class FailingReactRuntime(ReactRuntime):
    async def invoke(self, _call):
        raise RuntimeError("patch rejected")


class ErrorResultReactRuntime(ReactRuntime):
    async def invoke(self, _call):
        return [
            {
                "type": "text",
                "text": "[1201] invalid patch: hunk line counts do not match header",
            }
        ]


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
                AIMessage(content=(
                    "BEGIN STEP_COMPLETION_PROGRESS\nALL_COMPLETED=true\n"
                    "BEGIN COMPLETED_CRITERION\nNUMBER=1\nEVIDENCE=\"note written\"\n"
                    "END COMPLETED_CRITERION\nEND STEP_COMPLETION_PROGRESS"
                )),
                AIMessage(content="Done"),
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


def test_react_retries_with_failed_tool_message_at_the_end_of_context():
    class FailingObservationModel:
        def __init__(self):
            self.calls = []

        async def ainvoke(self, _messages):
            self.calls.append(_messages)
            return AIMessage(
                content=_observation(1, "write-1").replace(
                    'TEXT="write completed"', 'TEXT="patch rejected"'
                )
            )

    async def run():
        model = ReactModel()
        model.responses = iter(
            (
                AIMessage(
                    content="",
                    tool_calls=[{"name": "apply_patch", "args": {}, "id": "write-1"}],
                ),
                AIMessage(content="Recovered"),
            )
        )

        observation_model = FailingObservationModel()
        answer = await ReactMode(
            model,
            FailingReactRuntime(),
            context_policy=RuntimeContextPolicy(observation_model),
        ).run("Apply the patch")

        assert answer == ExecutionAnswer("Recovered", "completed")
        assert observation_model.calls == []
        failed_message = model.requests[1][-1]
        assert isinstance(failed_message, ToolMessage)
        assert failed_message.status == "error"
        assert failed_message.tool_call_id == "write-1"
        assert "patch rejected" in failed_message.content

    asyncio.run(run())


def test_react_retries_with_mcp_text_error_message_at_the_end_of_context():
    async def run():
        model = ReactModel()
        model.responses = iter(
            (
                AIMessage(
                    content="",
                    tool_calls=[{"name": "apply_patch", "args": {}, "id": "write-1"}],
                ),
                AIMessage(content="Recovered"),
            )
        )

        answer = await ReactMode(
            model,
            ErrorResultReactRuntime(),
            context_policy=RuntimeContextPolicy(BlockingObservationModel()),
        ).run("Apply the patch")

        assert answer == ExecutionAnswer("Recovered", "completed")
        failed_message = model.requests[1][-1]
        assert isinstance(failed_message, ToolMessage)
        assert failed_message.status == "error"
        assert failed_message.tool_call_id == "write-1"
        assert "invalid patch" in failed_message.content

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
        assert '"tool_call_id": "write-1"' in model.requests[1][-1].content
        assert '"raw_result": "wrote note"' in model.requests[1][-1].content

        observation_model.release.set()
        await asyncio.sleep(0)

    asyncio.run(run())


def test_executor_retries_with_failed_tool_message_at_the_end_of_context(monkeypatch):
    def reject_patch() -> str:
        """Reject the patch for retry testing."""
        raise RuntimeError("patch rejected")

    async def load_tools(_session):
        return [StructuredTool.from_function(reject_patch, name="apply_patch")]

    class FailingObservationModel:
        def __init__(self):
            self.calls = []

        async def ainvoke(self, _messages):
            self.calls.append(_messages)
            return AIMessage(
                content=_observation(1, "write-1").replace(
                    'TEXT="write completed"', 'TEXT="patch rejected"'
                )
            )

    async def run():
        model = ExecutorModel()
        model.responses = iter(
            (
                AIMessage(
                    content="",
                    tool_calls=[{"name": "apply_patch", "args": {}, "id": "write-1"}],
                ),
                AIMessage(content="Recovered"),
                AIMessage(content=(
                    "BEGIN STEP_COMPLETION_PROGRESS\nALL_COMPLETED=true\n"
                    "BEGIN COMPLETED_CRITERION\nNUMBER=1\nEVIDENCE=\"recovered\"\n"
                    "END COMPLETED_CRITERION\nEND STEP_COMPLETION_PROGRESS"
                )),
                AIMessage(content="Recovered"),
            )
        )
        monkeypatch.setattr("agent.executor.MultiServerMCPClient", Client)
        monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)

        observation_model = FailingObservationModel()
        async with Executor(
            model=model,
            context_policy=RuntimeContextPolicy(observation_model),
        ) as executor:
            outcome = await executor.execute(_step_state(), 1, "write")

        assert outcome.execution.result == "Recovered"
        assert observation_model.calls == []
        failed_message = model.requests[1][-1]
        assert isinstance(failed_message, ToolMessage)
        assert failed_message.status == "error"
        assert failed_message.tool_call_id == "write-1"
        assert "patch rejected" in failed_message.content

    asyncio.run(run())


def test_executor_retries_with_mcp_text_error_message_at_the_end_of_context(monkeypatch):
    def reject_patch() -> list[dict[str, str]]:
        """Return Atom's structured patch-validation failure."""
        return [
            {
                "type": "text",
                "text": "[1201] invalid patch: hunk line counts do not match header",
            }
        ]

    async def load_tools(_session):
        return [StructuredTool.from_function(reject_patch, name="apply_patch")]

    async def run():
        model = ExecutorModel()
        model.responses = iter(
            (
                AIMessage(
                    content="",
                    tool_calls=[{"name": "apply_patch", "args": {}, "id": "write-1"}],
                ),
                AIMessage(content="Recovered"),
                AIMessage(content=(
                    "BEGIN STEP_COMPLETION_PROGRESS\nALL_COMPLETED=true\n"
                    "BEGIN COMPLETED_CRITERION\nNUMBER=1\nEVIDENCE=\"recovered\"\n"
                    "END COMPLETED_CRITERION\nEND STEP_COMPLETION_PROGRESS"
                )),
                AIMessage(content="Recovered"),
            )
        )
        monkeypatch.setattr("agent.executor.MultiServerMCPClient", Client)
        monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)

        async with Executor(model=model) as executor:
            outcome = await executor.execute(_step_state(), 1, "write")

        assert outcome.execution.result == "Recovered"
        failed_message = model.requests[1][-1]
        assert isinstance(failed_message, ToolMessage)
        assert failed_message.status == "error"
        assert failed_message.tool_call_id == "write-1"
        assert failed_message.content[0]["text"].startswith("[1201] invalid patch")

    asyncio.run(run())
