import asyncio

import pytest

from agent import ExecutionAnswer, RoutedExecutionAnswer, TaskAnalysis, TaskRouter, route
from agent.durable import _routed_result, _task_router
from agent.task_analyzer import TaskAnalysisValidationError


def analysis(**overrides):
    values = {
        "task_type": "research",
        "goal_clarity": 0.9,
        "needs_tools": True,
        "expected_steps": 4,
        "expected_horizon": "medium",
        "reasoning_summary": "The task needs investigation.",
    }
    values.update(overrides)
    return TaskAnalysis(**values)


@pytest.mark.parametrize(
    ("characteristics", "expected"),
    [
        ({"needs_tools": False}, "direct"),
        ({"expected_steps": 1}, "tool_agent"),
        ({"expected_steps": 2}, "react"),
        ({"expected_horizon": "long"}, "plan_execute"),
    ],
)
def test_route_applies_the_ordered_execution_mode_policy(characteristics, expected):
    assert route(analysis(**characteristics)) == expected


def test_route_preserves_early_precedence_over_plan_execute_conditions():
    assert route(
        analysis(
            needs_tools=False,
            expected_horizon="long",
        )
    ) == "direct"
    assert route(
        analysis(
            expected_steps=1,
            expected_horizon="long",
        )
    ) == "tool_agent"


class ControlledAnalyzer:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.goals = []

    async def run(self, goal):
        self.goals.append(goal)
        if self.error:
            raise self.error
        return self.result


class ControlledMode:
    def __init__(self, answer):
        self.answer = answer
        self.goals = []

    async def run(self, goal):
        self.goals.append(goal)
        return self.answer


def test_router_analyzes_selects_one_mode_and_preserves_completed_answer():
    analyzer = ControlledAnalyzer(analysis(needs_tools=False))
    mode = ControlledMode(ExecutionAnswer("Done.", "completed"))
    selected = []

    def factory(name, task_analysis):
        selected.append((name, task_analysis))
        return mode

    result = asyncio.run(TaskRouter(analyzer, factory).run("Answer the question"))

    assert result == RoutedExecutionAnswer(
        ExecutionAnswer("Done.", "completed"), "direct", analysis(needs_tools=False)
    )
    assert analyzer.goals == ["Answer the question"]
    assert selected == [("direct", analysis(needs_tools=False))]
    assert mode.goals == ["Answer the question"]


def test_router_falls_back_to_plan_execute_once_with_safe_analysis_error():
    analyzer = ControlledAnalyzer(error=RuntimeError("provider secret"))
    mode = ControlledMode(ExecutionAnswer(None, "failed", error="selected mode failed"))
    selected = []

    def factory(name, task_analysis):
        selected.append((name, task_analysis))
        return mode

    result = asyncio.run(TaskRouter(analyzer, factory).run("Investigate it"))

    assert result.execution == ExecutionAnswer(None, "failed", error="selected mode failed")
    assert result.execution_mode == "plan_execute"
    assert result.analysis is None
    assert result.analysis_error == "task analysis failed"
    assert selected == [("plan_execute", None)]
    assert mode.goals == ["Investigate it"]
    assert "provider secret" not in result.analysis_error


@pytest.mark.parametrize(
    "error",
    [
        TaskAnalysisValidationError("malformed analysis"),
        ValueError("provider key=secret"),
    ],
)
def test_router_treats_invalid_analysis_as_a_conservative_plan_execute_fallback(error):
    analyzer = ControlledAnalyzer(error=error)
    mode = ControlledMode(ExecutionAnswer("Planned.", "completed"))
    selected = []

    def factory(name, task_analysis):
        selected.append((name, task_analysis))
        return mode

    result = asyncio.run(TaskRouter(analyzer, factory).run("Handle it"))

    assert result.execution == ExecutionAnswer("Planned.", "completed")
    assert result.execution_mode == "plan_execute"
    assert result.analysis is None
    assert result.analysis_error == "task analysis failed"
    assert "secret" not in result.analysis_error
    assert selected == [("plan_execute", None)]


def test_router_does_not_run_another_mode_after_selected_mode_fails():
    analyzer = ControlledAnalyzer(analysis(needs_tools=True, expected_steps=1))
    calls = []

    def factory(name, task_analysis):
        calls.append((name, task_analysis))
        return ControlledMode(ExecutionAnswer(None, "failed", error="no"))

    result = asyncio.run(TaskRouter(analyzer, factory).run("Use the tool"))

    assert result.execution_mode == "tool_agent"
    assert result.execution.status == "failed"
    assert calls == [("tool_agent", analysis(needs_tools=True, expected_steps=1))]


def test_router_passes_react_analysis_with_ordered_criteria_to_factory_and_result():
    task_analysis = analysis(completion_criteria=("Inspect the evidence.", "Report the cause."))
    analyzer = ControlledAnalyzer(task_analysis)
    mode = ControlledMode(ExecutionAnswer("Cause reported.", "completed"))
    received = []

    def factory(name, selected_analysis):
        received.append((name, selected_analysis))
        return mode

    result = asyncio.run(TaskRouter(analyzer, factory).run("Investigate the issue"))

    assert received == [("react", task_analysis)]
    assert received[0][1].completion_criteria == ("Inspect the evidence.", "Report the cause.")
    assert result.analysis is task_analysis
    assert _routed_result("run-id", result)["analysis"]["completion_criteria"] == (
        "Inspect the evidence.",
        "Report the cause.",
    )


def test_run_entry_router_uses_the_existing_mode_factory(monkeypatch):
    analyzer = ControlledAnalyzer(analysis(needs_tools=False))
    durable_agent = object()
    configuration = object()
    trace = object()
    selected = []
    mode = ControlledMode(ExecutionAnswer("Done.", "completed"))

    def factory(name, **kwargs):
        selected.append((name, kwargs))
        return mode

    monkeypatch.setattr("agent.durable.create_execution_mode", factory)

    result = asyncio.run(
        _task_router(
            analyzer,
            run_id="run-entry-id",
            configuration=configuration,
            durable_agent=durable_agent,
            tool_cwd="/workspace/project",
            trace=trace,
        ).run("Answer the question")
    )

    assert result.execution == ExecutionAnswer("Done.", "completed")
    assert selected[0][0] == "direct"
    assert selected[0][1]["configuration"] is configuration
    assert selected[0][1]["durable_agent"] is durable_agent
    assert selected[0][1]["run_id"] == "run-entry-id"
    assert selected[0][1]["tool_runtime"]._tool_cwd == "/workspace/project"


def test_run_entry_router_passes_react_criteria_to_execution_mode(monkeypatch):
    criteria = ("Inspect the evidence.", "Report the cause.")
    analyzer = ControlledAnalyzer(analysis(completion_criteria=criteria))
    selected = []
    mode = ControlledMode(ExecutionAnswer("Cause reported.", "completed"))

    def factory(name, **kwargs):
        selected.append((name, kwargs))
        return mode

    monkeypatch.setattr("agent.durable.create_execution_mode", factory)

    result = asyncio.run(
        _task_router(
            analyzer,
            run_id="run-entry-id",
            configuration=object(),
            durable_agent=object(),
            tool_cwd="/workspace/project",
            trace=object(),
        ).run("Investigate the issue")
    )

    assert result.execution.status == "completed"
    assert selected[0][0] == "react"
    assert selected[0][1]["completion_criteria"] == criteria


def test_run_entry_plan_execute_reuses_the_cli_run_id():
    analyzer = ControlledAnalyzer(analysis(expected_horizon="long"))

    class DurableAgent:
        def __init__(self):
            self.calls = []

        async def run(self, run_id, state):
            self.calls.append((run_id, state))
            return {
                "status": "completed",
                "state": {
                    "goal": state.goal,
                    "plan_history": [
                        {
                            "revision": 1,
                            "goal": state.goal,
                            "steps": [
                                {
                                    "id": "inspect",
                                    "intent": "Inspect",
                                    "completion_criterion": "Inspected",
                                }
                            ],
                        }
                    ],
                    "step_executions": [
                        {
                            "revision": 1,
                            "step_id": "inspect",
                            "status": "completed",
                            "result": "Inspected.",
                            "error": None,
                        }
                    ],
                    "memory_summary": None,
                },
            }

    durable = DurableAgent()
    result = asyncio.run(
        _task_router(
            analyzer,
            configuration=object(),
            durable_agent=durable,
            tool_cwd="/workspace/project",
            trace=object(),
            run_id="run-entry-id",
        ).run("Inspect the project")
    )

    assert result.execution == ExecutionAnswer("Inspected.", "completed")
    assert durable.calls[0][0] == "run-entry-id"


def test_run_result_exposes_the_router_outcome_without_changing_execution_answer():
    routed = RoutedExecutionAnswer(
        ExecutionAnswer(None, "failed", error="direct model invocation failed"),
        "direct",
        None,
        "task analysis failed",
    )

    assert _routed_result("run-1", routed) == {
        "run_id": "run-1",
        "status": "failed",
        "execution_mode": "direct",
        "execution": {
            "answer": None,
            "status": "failed",
            "error": "direct model invocation failed",
        },
        "analysis": None,
        "analysis_error": "task analysis failed",
    }
