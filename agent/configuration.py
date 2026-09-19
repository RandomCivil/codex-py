"""Explicit YAML configuration for Planner, Executor, and Execution modes."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from llm.response_format import RESPONSE_FORMATS, ResponseFormat, require_response_format


class ConfigurationError(ValueError):
    """The explicit component provider configuration is invalid."""


@dataclass(frozen=True)
class ProviderConfiguration:
    base_url: str
    api_key: str
    model_name: str
    response_format: ResponseFormat | None


@dataclass(frozen=True)
class ComponentProviderConfiguration:
    planner: ProviderConfiguration
    executor: ProviderConfiguration
    task_analyzer: ProviderConfiguration
    runtime_context: ProviderConfiguration | None = None
    direct: ProviderConfiguration | None = None
    tool_agent: ProviderConfiguration | None = None
    react: ProviderConfiguration | None = None


_FIELDS = {"base_url", "api_key", "model_name", "response_format"}


def load_configuration(path: str | Path) -> ComponentProviderConfiguration:
    try:
        document = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError(f"cannot read configuration: {error}") from error

    if not isinstance(document, dict):
        raise ConfigurationError("configuration must be a YAML mapping")
    _check_keys(
        document,
        {"model", "planner", "executor", "task_analyzer", "runtime_context", "direct", "tool_agent", "react"},
        "configuration",
    )
    shared = _mapping(document.get("model"), "model")
    planner = _resolve(shared, document.get("planner"), "planner")
    executor = _resolve(shared, document.get("executor"), "executor")
    task_analyzer = _resolve(planner, document.get("task_analyzer"), "task_analyzer")
    runtime_context = (
        _resolve(shared, document["runtime_context"], "runtime_context")
        if "runtime_context" in document
        else None
    )
    direct = _resolve(shared, document.get("direct"), "direct")
    tool_agent = _resolve(shared, document.get("tool_agent"), "tool_agent")
    react = _resolve(shared, document.get("react"), "react")
    return ComponentProviderConfiguration(
        planner=_provider(planner, "planner", require_output_format=True),
        executor=_provider(executor, "executor", require_output_format=True),
        task_analyzer=_provider(task_analyzer, "task_analyzer", require_output_format=True),
        runtime_context=(
            _provider(runtime_context, "runtime_context", require_output_format=True)
            if runtime_context is not None
            else None
        ),
        direct=_provider(direct, "direct", require_output_format=False),
        tool_agent=_provider(tool_agent, "tool_agent", require_output_format=False),
        react=_provider(react, "react", require_output_format=True),
    )


def _resolve(shared: Mapping[str, Any], override: Any, name: str) -> dict[str, Any]:
    values = dict(shared)
    if override is not None:
        values.update(_mapping(override, name))
    return values


def _provider(
    values: Mapping[str, Any],
    name: str,
    *,
    require_output_format: bool,
) -> ProviderConfiguration:
    _check_keys(values, _FIELDS, name)
    required = ["base_url", "api_key", "model_name"]
    if require_output_format:
        required.append("response_format")
    missing = [field for field in required if field not in values]
    if missing:
        raise ConfigurationError(f"{name} is missing required field(s): {', '.join(missing)}")
    for field in ("base_url", "api_key", "model_name"):
        value = values[field]
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(f"{name}.{field} must be a non-empty string")
    response_format = None
    if "response_format" in values:
        try:
            response_format = require_response_format(values["response_format"])
        except ValueError as error:
            raise ConfigurationError(
                f"{name}.response_format must be one of: {', '.join(sorted(RESPONSE_FORMATS))}"
            ) from error
    return ProviderConfiguration(
        base_url=values["base_url"],
        api_key=values["api_key"],
        model_name=values["model_name"],
        response_format=response_format,
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{name} must be a YAML mapping")
    return value


def _check_keys(values: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(str(key) for key in set(values) - allowed)
    if unknown:
        raise ConfigurationError(f"{name} contains unknown field(s): {', '.join(unknown)}")
