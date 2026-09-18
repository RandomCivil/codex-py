from agent.configuration import load_configuration
import pytest


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
