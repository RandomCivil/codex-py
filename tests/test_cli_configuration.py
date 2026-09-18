import json
from pathlib import Path

from agent.cli import EXIT_CONFIGURATION, EXIT_INVALID_INVOCATION, main


def test_run_requires_an_explicit_configuration_file(monkeypatch, capsys):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")

    assert main(["run", "--goal", "Prepare release"]) == EXIT_INVALID_INVOCATION
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"


def test_run_rejects_invalid_configuration_before_agent_work(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    path = tmp_path / "agent.yaml"
    path.write_text("model:\n  response_format: json_schema\n")

    assert main(["run", "--goal", "Prepare release", "--config", str(path)]) == EXIT_CONFIGURATION
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "configuration"


def test_run_reports_non_scalar_response_format_as_a_configuration_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    path = tmp_path / "agent.yaml"
    path.write_text(
        "model:\n"
        "  base_url: https://provider.test\n"
        "  api_key: key\n"
        "  model_name: model\n"
        "  response_format: []\n"
    )

    assert main(["run", "--goal", "Prepare release", "--config", str(path)]) == EXIT_CONFIGURATION
    assert json.loads(capsys.readouterr().out)["status"] == "configuration"


def test_run_forwards_resolved_component_configurations(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    path = tmp_path / "agent.yaml"
    path.write_text(
        "model:\n"
        "  base_url: https://shared.example/v1\n"
        "  api_key: shared-key\n"
        "  model_name: shared-model\n"
        "  response_format: json_schema\n"
        "planner:\n"
        "  model_name: planner-model\n"
    )
    observed = {}

    def run(url, goal, **kwargs):
        observed["configuration"] = kwargs["configuration"]
        return {"status": "completed"}

    monkeypatch.setattr("agent.cli.run_agent", run)

    assert main(["run", "--goal", "Prepare release", "--config", str(path)]) == 0
    configuration = observed["configuration"]
    assert configuration.planner.model_name == "planner-model"
    assert configuration.executor.model_name == "shared-model"
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_migrate_remains_independent_of_model_configuration(monkeypatch, capsys):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    monkeypatch.setattr("agent.cli.migrate_database", lambda url: None)

    assert main(["migrate"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_resume_forwards_the_resolved_component_configurations(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CODEX_MYSQL_URL", "mysql://user:pass@localhost/db")
    path = tmp_path / "agent.yaml"
    path.write_text(
        "model:\n"
        "  base_url: https://shared.example/v1\n"
        "  api_key: shared-key\n"
        "  model_name: shared-model\n"
        "  response_format: json_object\n"
        "executor:\n"
        "  base_url: https://executor.example/v1\n"
    )
    observed = {}

    def resume(url, run_id, **kwargs):
        observed["configuration"] = kwargs["configuration"]
        return {"status": "completed", "run_id": run_id}

    monkeypatch.setattr("agent.cli.resume_agent", resume)

    assert main(["resume", "--run-id", "550e8400-e29b-41d4-a716-446655440000", "--config", str(path)]) == 0
    configuration = observed["configuration"]
    assert configuration.planner.model_name == "shared-model"
    assert configuration.planner.response_format == "json_object"
    assert configuration.executor.base_url == "https://executor.example/v1"
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_operator_documentation_describes_the_yaml_configuration_workflow():
    documentation = Path("README.md").read_text()

    assert "agent.yaml" in documentation
    assert "planner:" in documentation
    assert "executor:" in documentation
    assert "json_schema" in documentation
    assert "json_object" in documentation
    assert "文件权限" in documentation
    assert "OPENAI_BASE_URL" not in documentation
    assert "OPENAI_API_KEY" not in documentation
    assert "OPENAI_MODEL" not in documentation
