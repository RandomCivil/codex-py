import asyncio

from langgraph.checkpoint.memory import InMemorySaver

from agent.durable import DurableAgent
from agent.recovery import InMemoryStepRecoveryStore
from memory.state import AgentState, ContextUpdate, ExecutionOutcome, Plan, PlanStep, StepContext, StepExecution


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
