import pytest

from agent.configuration import ConfigurationError, load_configuration


def test_load_configuration_resolves_provider_defaults_without_output_mode(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        """
model:
  base_url: https://shared.example/v1
  api_key: shared-secret
  model_name: shared-model
direct:
  model_name: direct-model
tool_agent:
  base_url: https://tools.example/v1
runtime_context:
  base_url: https://context.example/v1
  model_name: context-model
"""
    )

    configuration = load_configuration(path)

    assert configuration.direct.model_name == "direct-model"
    assert configuration.direct.api_key == "shared-secret"
    assert configuration.tool_agent.base_url == "https://tools.example/v1"
    assert configuration.react.model_name == "shared-model"
    assert configuration.runtime_context.model_name == "context-model"
    assert not hasattr(configuration.planner, "response_format")
    assert configuration.planner.stream is False


def test_load_configuration_resolves_stream_boolean_overrides(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        """
model:
  base_url: https://provider.test
  api_key: key
  model_name: model
  stream: true
planner:
  stream: false
direct:
  stream: true
"""
    )

    configuration = load_configuration(path)

    assert configuration.planner.stream is False
    assert configuration.executor.stream is True
    assert configuration.task_analyzer.stream is False
    assert configuration.direct.stream is True


@pytest.mark.parametrize("value", ("true", 1, [], None))
def test_load_configuration_rejects_non_boolean_stream(tmp_path, value):
    path = tmp_path / "agent.yaml"
    path.write_text(
        "model:\n  base_url: https://provider.test\n  api_key: key\n"
        f"  model_name: model\n  stream: {value!r}\n"
    )

    with pytest.raises(ConfigurationError, match=r"planner\.stream must be a boolean"):
        load_configuration(path)


@pytest.mark.parametrize("value", ("json_schema", "json_object", [], None))
def test_load_configuration_rejects_removed_response_format_at_every_level(tmp_path, value):
    path = tmp_path / "agent.yaml"
    path.write_text(
        "model:\n  base_url: https://provider.test\n  api_key: key\n"
        f"  model_name: model\n  response_format: {value!r}\n"
    )

    with pytest.raises(ConfigurationError, match="response_format.*removed.*Line Protocol"):
        load_configuration(path)


def test_load_configuration_rejects_unknown_and_missing_fields(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text("model:\n  base_url: https://provider.test\n  typo: value\n")

    with pytest.raises(ConfigurationError, match="unknown field"):
        load_configuration(path)

    path.write_text("model:\n  base_url: https://provider.test\n  api_key: key\n")
    with pytest.raises(ConfigurationError, match="missing required"):
        load_configuration(path)
