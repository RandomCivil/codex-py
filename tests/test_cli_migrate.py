import json
import uuid

from agent.cli import EXIT_BUSY, EXIT_CONFIGURATION, main
from agent.registry import ConfigurationMismatchError, RunBusyError


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


def test_run_emits_a_uuidv4_agent_run_id(monkeypatch, capsys):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    monkeypatch.setattr("agent.cli.run_agent", lambda url, goal: {"status": "completed"})

    exit_code = main(["run", "--goal", "Prepare release"])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output["command"] == "run"
    assert output["status"] == "completed"
    assert uuid.UUID(output["run_id"]).version == 4


def test_run_forwards_cwd_to_agent(monkeypatch, capsys):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    calls = []

    def run(url, goal, cwd):
        calls.append((url, goal, cwd))
        return {"status": "completed"}

    monkeypatch.setattr("agent.cli.run_agent", run)

    assert main(["run", "--goal", "Prepare release", "--cwd", "/workspace/project"]) == 0
    assert calls == [("mysql://user:pass@localhost/db", "Prepare release", "/workspace/project")]


def test_resume_requires_a_run_id(monkeypatch, capsys):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")

    exit_code = main(["resume"])

    assert exit_code == 5
    assert json.loads(capsys.readouterr().out) == {
        "command": "resume",
        "status": "invalid",
        "error": "resume requires --run-id and does not accept --goal",
    }


def test_parser_errors_are_stable_json(monkeypatch, capsys):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")

    assert main([]) == 5
    assert json.loads(capsys.readouterr().out) == {
        "command": None,
        "status": "invalid",
        "error": "the following arguments are required: command",
    }


def test_resume_configuration_mismatch_is_a_configuration_result(monkeypatch, capsys):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    monkeypatch.setattr("agent.cli.resume_agent", lambda url, run_id: (_ for _ in ()).throw(ConfigurationMismatchError("configuration changed")))

    assert main(["resume", "--run-id", str(uuid.uuid4())]) == EXIT_CONFIGURATION
    assert json.loads(capsys.readouterr().out)["status"] == "configuration"


def test_competing_resume_is_a_busy_result(monkeypatch, capsys):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    monkeypatch.setattr("agent.cli.resume_agent", lambda url, run_id: (_ for _ in ()).throw(RunBusyError("run busy")))

    assert main(["resume", "--run-id", str(uuid.uuid4())]) == EXIT_BUSY
    assert json.loads(capsys.readouterr().out) == {
        "command": "resume",
        "status": "busy",
        "error": "run busy",
    }


def test_resume_accepts_an_explicit_recovery_decision(monkeypatch, capsys):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    calls = []

    def resume(url, run_id, recovery):
        calls.append(recovery)
        return {"status": "blocked", "run_id": run_id}

    monkeypatch.setattr("agent.cli.resume_agent", resume)

    assert main(["resume", "--run-id", str(uuid.uuid4()), "--recovery", "abort"]) == 1
    assert calls == ["abort"]
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"
