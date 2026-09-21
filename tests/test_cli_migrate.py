import json
import uuid

import pytest

from agent.cli import EXIT_BUSY, EXIT_CONFIGURATION, main
from agent.registry import ConfigurationMismatchError, RunBusyError


def _configuration_file(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        "model:\n"
        "  base_url: https://provider.test/v1\n"
        "  api_key: key\n"
        "  model_name: model\n"
    )
    return str(path)


def test_migrate_requires_codex_mysql_url(monkeypatch, capsys):
    monkeypatch.delenv("CODEX_MYSQL_URL", raising=False)

    exit_code = main(["migrate"])

    assert exit_code == EXIT_CONFIGURATION
    assert json.loads(capsys.readouterr().out) == {
        "command": "migrate",
        "status": "configuration",
        "error": "CODEX_MYSQL_URL is required",
    }


def test_migrate_emits_completed_result(monkeypatch, capsys):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    monkeypatch.setattr("agent.cli.migrate_database", lambda url: None)

    exit_code = main(["migrate"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "command": "migrate",
        "status": "completed",
    }


def test_run_emits_a_uuidv4_agent_run_id(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    monkeypatch.setattr("agent.cli.run_agent", lambda url, goal, **kwargs: {"status": "completed"})

    exit_code = main(["run", "--goal", "Prepare release", "--config", _configuration_file(tmp_path)])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["command"] == "run"
    assert output["status"] == "completed"
    assert uuid.UUID(output["run_id"]).version == 4


def test_run_forwards_cwd_to_agent(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    calls = []

    def run(url, goal, cwd, **kwargs):
        calls.append((url, goal, cwd))
        return {"status": "completed"}

    monkeypatch.setattr("agent.cli.run_agent", run)

    assert main(["run", "--goal", "Prepare release", "--cwd", "/workspace/project", "--config", _configuration_file(tmp_path)]) == 0
    assert calls == [("mysql://user:pass@localhost/db", "Prepare release", "/workspace/project")]


def test_run_forwards_log_level_to_agent(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    calls = []

    def run(url, goal, log_level, **kwargs):
        calls.append((url, goal, log_level))
        return {"status": "completed"}

    monkeypatch.setattr("agent.cli.run_agent", run)

    assert main(["run", "--goal", "Prepare release", "--log-level", "error", "--config", _configuration_file(tmp_path)]) == 0
    assert calls == [("mysql://user:pass@localhost/db", "Prepare release", "error")]


def test_resume_requires_a_run_id(monkeypatch, capsys):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")

    exit_code = main(["resume"])

    assert exit_code == 5
    assert json.loads(capsys.readouterr().out) == {
        "command": "resume",
        "status": "invalid",
        "error": "resume requires --run-id and --config and does not accept --goal",
    }


def test_parser_errors_are_stable_json(monkeypatch, capsys):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")

    assert main([]) == 5
    assert json.loads(capsys.readouterr().out) == {
        "command": None,
        "status": "invalid",
        "error": "the following arguments are required: command",
    }


def test_resume_configuration_mismatch_is_a_configuration_result(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    monkeypatch.setattr("agent.cli.resume_agent", lambda url, run_id, **kwargs: (_ for _ in ()).throw(ConfigurationMismatchError("configuration changed")))

    assert main(["resume", "--run-id", str(uuid.uuid4()), "--config", _configuration_file(tmp_path)]) == EXIT_CONFIGURATION
    assert json.loads(capsys.readouterr().out)["status"] == "configuration"


def test_competing_resume_is_a_busy_result(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    monkeypatch.setattr("agent.cli.resume_agent", lambda url, run_id, **kwargs: (_ for _ in ()).throw(RunBusyError("run busy")))

    assert main(["resume", "--run-id", str(uuid.uuid4()), "--config", _configuration_file(tmp_path)]) == EXIT_BUSY
    assert json.loads(capsys.readouterr().out) == {
        "command": "resume",
        "status": "busy",
        "error": "run busy",
    }


def test_resume_accepts_an_explicit_recovery_decision(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    calls = []

    def resume(url, run_id, recovery, **kwargs):
        calls.append(recovery)
        return {"status": "blocked", "run_id": run_id}

    monkeypatch.setattr("agent.cli.resume_agent", resume)

    assert main(["resume", "--run-id", str(uuid.uuid4()), "--recovery", "abort", "--config", _configuration_file(tmp_path)]) == 1
    assert calls == ["abort"]
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_resume_rejects_redundant_retry_recovery_selection(monkeypatch, capsys):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")

    assert main(["resume", "--run-id", str(uuid.uuid4()), "--recovery", "retry"]) == 5
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "invalid"


def test_conversation_run_emits_current_turn_result(monkeypatch, capsys, tmp_path):
    from agent.conversation import ConversationResult

    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    monkeypatch.setattr(
        "agent.cli.run_mysql_conversation",
        lambda *args, **kwargs: ConversationResult("conv", 1, "run", "direct", "completed", "Hello"),
    )

    assert main(["conversation", "run", "--input", "Hello", "--config", _configuration_file(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "command": "conversation",
        "conv_id": "conv",
        "sequence": 1,
        "run_id": "run",
        "execution_mode": "direct",
        "status": "completed",
        "answer": "Hello",
        "error": None,
    }


def test_conversation_run_leaves_mode_selection_to_the_task_analyzer(monkeypatch, capsys, tmp_path):
    from agent.conversation import ConversationResult

    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    calls = []

    def run(url, **kwargs):
        calls.append((url, kwargs))
        return ConversationResult("conv", 2, "run", "react", "completed", "Answer")

    monkeypatch.setattr("agent.cli.run_mysql_conversation", run)

    assert main(
        [
            "conversation",
            "run",
            "--input",
            "Continue",
            "--conv-id",
            "550e8400-e29b-41d4-a716-446655440000",
            "--cwd",
            "/workspace/project",
            "--config",
            _configuration_file(tmp_path),
        ]
    ) == 0

    assert calls[0][0] == "mysql://user:pass@localhost/db"
    assert "execution_mode" not in calls[0][1]
    assert calls[0][1]["conv_id"] == "550e8400-e29b-41d4-a716-446655440000"
    assert calls[0][1]["cwd"] == "/workspace/project"
    assert json.loads(capsys.readouterr().out)["execution_mode"] == "react"


def test_conversation_run_forwards_log_level(monkeypatch, capsys, tmp_path):
    from agent.conversation import ConversationResult

    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    calls = []

    def run(url, **kwargs):
        calls.append((url, kwargs))
        return ConversationResult("conv", 1, "run", "direct", "completed", "Answer")

    monkeypatch.setattr("agent.cli.run_mysql_conversation", run)

    assert main(
        [
            "conversation",
            "run",
            "--input",
            "Hello",
            "--log-level",
            "error",
            "--config",
            _configuration_file(tmp_path),
        ]
    ) == 0

    assert calls[0][1]["log_level"] == "error"


def test_conversation_resume_emits_only_the_recovered_turn(monkeypatch, capsys, tmp_path):
    from agent.conversation import ConversationResult

    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    calls = []

    def resume(url, **kwargs):
        calls.append((url, kwargs))
        return ConversationResult(
            "550e8400-e29b-41d4-a716-446655440000",
            3,
            "550e8400-e29b-41d4-a716-446655440001",
            "plan_execute",
            "blocked",
            error="recovery aborted",
        )

    monkeypatch.setattr("agent.cli.resume_mysql_conversation", resume)

    assert main(
        [
            "conversation",
            "resume",
            "--conv-id",
            "550e8400-e29b-41d4-a716-446655440000",
            "--recovery",
            "abort",
            "--config",
            _configuration_file(tmp_path),
        ]
    ) == 1

    assert len(calls) == 1
    assert calls[0][0] == "mysql://user:pass@localhost/db"
    assert calls[0][1]["conv_id"] == "550e8400-e29b-41d4-a716-446655440000"
    assert calls[0][1]["recovery"] == "abort"
    assert calls[0][1]["configuration"] is not None
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "command": "conversation",
        "conv_id": "550e8400-e29b-41d4-a716-446655440000",
        "sequence": 3,
        "run_id": "550e8400-e29b-41d4-a716-446655440001",
        "execution_mode": "plan_execute",
        "status": "blocked",
        "answer": None,
        "error": "recovery aborted",
    }


def test_conversation_show_emits_complete_history(monkeypatch, capsys):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    monkeypatch.setattr("agent.cli.show_mysql_conversation", lambda url, conv_id: [{"sequence": 1}])
    conv_id = str(uuid.uuid4())

    assert main(["conversation", "show", "--conv-id", conv_id]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "command": "conversation",
        "conv_id": conv_id,
        "history": [{"sequence": 1}],
    }
