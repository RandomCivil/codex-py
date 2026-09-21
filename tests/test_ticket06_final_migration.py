import inspect

import pytest

from agent.configuration import load_configuration
from agent.registry import configuration_snapshot
from llm import LLM


def test_configuration_has_no_structured_output_mode(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        """
model:
  base_url: https://provider.test/v1
  api_key: secret
  model_name: model
"""
    )

    configuration = load_configuration(path)

    assert not hasattr(configuration.planner, "response_format")
    assert configuration.planner.model_name == "model"


def test_legacy_response_format_is_rejected_as_removed(tmp_path):
    path = tmp_path / "agent.yaml"
    path.write_text(
        """
model:
  base_url: https://provider.test/v1
  api_key: secret
  model_name: model
  response_format: json_schema
"""
    )

    with pytest.raises(ValueError, match="response_format.*removed.*Line Protocol"):
        load_configuration(path)


def test_configuration_snapshot_omits_response_format():
    snapshot = configuration_snapshot(
        planner={"base_url": "https://planner.test", "model_name": "planner"},
        executor={"base_url": "https://executor.test", "model_name": "executor"},
        mcp={},
    )

    assert snapshot["planner"] == {
        "base_url": "https://planner.test",
        "model_name": "planner",
    }
    assert "response_format" not in repr(snapshot)


def test_llm_has_no_output_format_api():
    assert "response_format" not in inspect.signature(LLM).parameters
    assert "text_format" not in inspect.signature(LLM.stream_events).parameters
