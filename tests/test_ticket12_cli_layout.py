import json

from agent.cli import main
from agent.registry import configuration_fingerprint, configuration_snapshot


def _config(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        "model:\n  base_url: https://provider.test/v1\n  api_key: key\n  model_name: model\n"
    )
    return str(path)


def test_run_forwards_runtime_context_layout(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    observed = {}
    monkeypatch.setattr(
        "agent.cli.run_agent",
        lambda url, goal, **kwargs: (observed.update(kwargs) or {"status": "completed"}),
    )

    assert main(["run", "--goal", "x", "--config", _config(tmp_path), "--runtime-context-layout", "messages"]) == 0
    assert observed["runtime_context_layout"] == "messages"
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_resume_omitting_layout_uses_grouped(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    observed = {}
    monkeypatch.setattr(
        "agent.cli.resume_agent",
        lambda url, run_id, **kwargs: (observed.update(kwargs) or {"status": "completed", "run_id": run_id}),
    )

    assert main(["resume", "--run-id", "550e8400-e29b-41d4-a716-446655440000", "--config", _config(tmp_path)]) == 0
    assert observed["runtime_context_layout"] == "grouped"
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_runtime_context_layout_is_part_of_configuration_identity():
    base = configuration_snapshot(planner={"base_url": "u", "model_name": "m"}, mcp={})
    messages = configuration_snapshot(
        planner={"base_url": "u", "model_name": "m"}, mcp={}, runtime_context_layout="messages"
    )
    assert base["runtime_context_layout"] == "grouped"
    assert messages["runtime_context_layout"] == "messages"
    assert configuration_fingerprint(base) != configuration_fingerprint(messages)
