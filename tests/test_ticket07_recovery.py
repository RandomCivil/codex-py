import asyncio
import json

from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from agent.durable import DurableAgent
from agent.recovery import InMemoryStepRecoveryStore
from agent.executor import Executor
from agent.runtime_context import RuntimeContextPolicy
from memory.state import AgentState, ContextUpdate, ExecutionOutcome, Plan, PlanStep, StepContext, StepExecution


class _ObservationModel:
    def __init__(self):
        self.requests = []

    def bind(self, **_kwargs):
        return self

    async def ainvoke(self, messages):
        self.requests.append(messages)
        payload = json.loads(messages[1].content)
        call_id = payload["calls"][0]["id"]
        return AIMessage(
            content="\n".join(
                (
                    "BEGIN OBSERVATION",
                    f"ROUND={payload['round']}",
                    "AFFECTS_CURRENT_DECISION=true",
                    'AFFECTED_TARGETS="durable_state"',
                    "BEGIN EVIDENCE",
                    'CATEGORY="confirmed_facts"',
                    'TEXT="the note was written"',
                    f"TOOL_CALL_ID={json.dumps(call_id)}",
                    "END EVIDENCE",
                    "END OBSERVATION",
                )
            )
        )


class _ExecutorModel:
    def __init__(self):
        self.requests = []
        self.responses = iter(
            (
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {"path": "note.md"},
                            "id": "write-1",
                        }
                    ],
                ),
                AIMessage(content=(
                    "BEGIN STEP_COMPLETION_PROGRESS\nALL_COMPLETED=true\n"
                    "BEGIN COMPLETED_CRITERION\nNUMBER=1\nEVIDENCE=\"the note was written\"\n"
                    "END COMPLETED_CRITERION\nEND STEP_COMPLETION_PROGRESS"
                )),
                AIMessage(content="Written"),
            )
        )

    def bind_tools(self, _tools):
        return self

    async def ainvoke(self, messages):
        self.requests.append(messages)
        return next(self.responses)


class _Session:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class _Client:
    def __init__(self, _config):
        pass

    def session(self, _name):
        return _Session()


def test_executor_observation_receives_the_current_plan_step_decision_summary(monkeypatch):
    def write_file(path: str) -> str:
        """Write the release note."""
        return f"wrote {path}"

    async def load_tools(_session):
        return [StructuredTool.from_function(write_file, name="write_file")]

    async def scenario():
        observation_model = _ObservationModel()
        executor_model = _ExecutorModel()
        monkeypatch.setattr("agent.executor.MultiServerMCPClient", _Client)
        monkeypatch.setattr("agent.executor.load_mcp_tools", load_tools)
        state = AgentState(
            "Prepare release",
            plan_history=(
                Plan(1, "Prepare release", (PlanStep("publish", "Publish", "Released"),)),
            ),
        )
        async with Executor(
            model=executor_model,
            context_policy=RuntimeContextPolicy(observation_model),
        ) as executor:
            outcome = await executor.execute(state, 1, "publish")
        await asyncio.sleep(0.01)
        return outcome, observation_model, executor_model, state

    outcome, observation_model, executor_model, state = asyncio.run(scenario())

    assert outcome.execution.status == "completed", outcome.execution.error
    payload = json.loads(observation_model.requests[0][1].content)
    assert payload["decision_summary"] == {
        "goal": "Prepare release",
        "execution_mode": "plan_execute",
        "plan_step": {
            "revision": 1,
            "id": "publish",
            "intent": "Publish",
            "completion_criterion": "Released",
        },
        "completion_criterion": "Released",
        "tool": {"name": "write_file", "arguments": {"path": "note.md"}},
    }
    assert '"raw_result": "wrote note.md"' in executor_model.requests[1][-1].content
    assert outcome.context_update == ContextUpdate()
    assert state.goal == "Prepare release"


def test_ordinary_resume_starts_a_fresh_recovery_attempt_with_new_identity():
    plan = Plan(1, "Prepare release", (PlanStep("publish", "Publish", "Released"),))
    state = AgentState(
        "Prepare release",
        plan_history=(plan,),
        step_executions=(),
    )

    class Executor:
        def __init__(self, fail=False):
            self.calls = []
            self.fail = fail

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def execute(self, state, revision, step_id, step_context=None, *, recovery=False, attempt=None):
            self.calls.append((recovery, attempt, step_context))
            if self.fail:
                raise RuntimeError("process stopped")
            return ExecutionOutcome(
                StepExecution(revision, step_id, "completed", result="Released"),
                ContextUpdate(),
            )

    async def scenario():
        saver = InMemorySaver()
        store = InMemoryStepRecoveryStore()
        first = Executor(fail=True)
        try:
            await DurableAgent(object(), first, saver, recovery_store=store).run("run-1", state)
        except RuntimeError:
            pass
        executor = Executor()
        result = await DurableAgent(object(), executor, saver, recovery_store=store).run("run-1")
        return result, executor, await store.history("run-1")

    result, executor, history = asyncio.run(scenario())

    assert result["status"] == "completed"
    assert executor.calls[0][0:2] == (True, 2)
    assert [(item.attempt, item.execution.status) for item in history] == [
        (1, "running"),
        (1, "interrupted"),
        (2, "running"),
        (2, "completed"),
    ]


def test_resume_passes_reconstructed_completed_step_context_to_recovery_executor():
    first_plan = Plan(1, "Prepare release", (PlanStep("z-complete", "Inspect", "Inspected"),))
    second_plan = Plan(2, "Prepare release", (PlanStep("a-recover", "Publish", "Released"),))
    checkpoint_state = AgentState(
        "Prepare release",
        plan_history=(first_plan, second_plan),
        step_executions=(
            StepExecution(1, "z-complete", "completed", result="Inspected"),
            StepExecution(2, "a-recover", "interrupted"),
        ),
    )

    class Executor:
        def __init__(self):
            self.contexts = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def execute(self, state, revision, step_id, step_context=None, *, recovery=False, attempt=None):
            self.contexts.append((step_context, recovery, attempt))
            return ExecutionOutcome(StepExecution(revision, step_id, "completed", result="Released"), ContextUpdate())

    async def scenario():
        saver = InMemorySaver()
        store = InMemoryStepRecoveryStore()
        await store.record("run-2", checkpoint_state.step_executions[0], context_update=ContextUpdate(files_read=("README.md",)), version=1)
        await store.record("run-2", checkpoint_state.step_executions[1], version=2)
        await DurableAgent(object(), Executor(), saver)._graph().ainvoke(
            {"agent_state": checkpoint_state, "status": "interrupted", "persistence_version": 2},
            config={"configurable": {"thread_id": "run-2"}},
        )
        executor = Executor()
        result = await DurableAgent(object(), executor, saver, recovery_store=store).run("run-2")
        return result, executor

    result, executor = asyncio.run(scenario())

    assert result["status"] == "completed"
    assert executor.contexts == [(StepContext(files_read=("README.md",)), True, 2)]
