from agent.configuration import load_configuration
import pytest


def test_load_configuration_resolves_provider_defaults_for_every_execution_mode(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        """
model:
  base_url: https://shared.example/v1
  api_key: shared-secret
  model_name: shared-model
  response_format: json_object
direct:
  model_name: direct-model
tool_agent:
  base_url: https://tools.example/v1
react:
  response_format: json_schema
"""
    )

    configuration = load_configuration(path)

    assert configuration.direct.model_name == "direct-model"
    assert configuration.direct.api_key == "shared-secret"
    assert configuration.direct.response_format == "json_object"
    assert configuration.tool_agent.base_url == "https://tools.example/v1"
    assert configuration.tool_agent.model_name == "shared-model"
    assert configuration.react.response_format == "json_schema"


def test_load_configuration_resolves_shared_defaults_and_component_overrides(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        """
model:
  base_url: https://shared.example/v1
  api_key: shared-secret
  model_name: shared-model
  response_format: json_schema
planner:
  model_name: planner-model
executor:
  response_format: json_object
"""
    )

    configuration = load_configuration(path)

    assert configuration.planner.base_url == "https://shared.example/v1"
    assert configuration.planner.api_key == "shared-secret"
    assert configuration.planner.model_name == "planner-model"
    assert configuration.planner.response_format == "json_schema"
    assert configuration.executor.base_url == "https://shared.example/v1"
    assert configuration.executor.api_key == "shared-secret"
    assert configuration.executor.model_name == "shared-model"
    assert configuration.executor.response_format == "json_object"


def test_load_configuration_requires_structured_output_only_for_react(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        """
model:
  base_url: https://shared.example/v1
  api_key: shared-secret
  model_name: shared-model
planner:
  response_format: json_schema
executor:
  response_format: json_object
direct:
  model_name: direct-model
tool_agent:
  model_name: tool-model
"""
    )

    with pytest.raises(ValueError, match="react.*response_format"):
        load_configuration(path)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("model:\n  base_url: https://provider.test\n", "missing required field"),
        (
            "model:\n  base_url: https://provider.test\n  api_key: key\n  model_name: model\n  response_format: xml\n",
            "response_format must be one of",
        ),
        (
            "model:\n  base_url: https://provider.test\n  api_key: key\n  model_name: model\n  response_format: []\n",
            "response_format must be one of",
        ),
        (
            "model:\n  base_url: https://provider.test\n  api_key: key\n  model_name: model\n  response_format: json_schema\nplanner:\n  typo: value\n",
            "unknown field",
        ),
    ],
)
def test_load_configuration_rejects_invalid_documents(tmp_path, document, message):
    path = tmp_path / "agent.yaml"
    path.write_text(document)

    with pytest.raises(ValueError, match=message):
        load_configuration(path)
