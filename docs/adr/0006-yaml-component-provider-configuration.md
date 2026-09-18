# YAML component Provider configuration

**Status: accepted.** CLI `run` and `resume` will require an explicit `--config` path to a YAML file; environment variables are not a model-configuration source and `migrate` does not read this file. The document supplies a shared `model` default and optional `planner`, `executor`, `direct`, `tool_agent`, and `react` overrides for `base_url`, plaintext `api_key`, `model_name`, and Structured-output mode. Planner and Executor must resolve to complete Provider configurations; Direct and tool-agent require endpoint, credential, and model; ReAct additionally requires Structured-output mode. Invalid YAML, unknown fields, invalid values, or missing effective fields are configuration errors before any model or tool work.

`json_schema` and `json_object` are supported Structured-output modes for the Planner's Plan response and Executor's final completion receipt. `json_object` retains the existing local strict validation; it only removes provider-enforced schema validation. The Executor uses its own effective Provider configuration for all of its model requests, including tool-selection requests, but its MCP tool calls remain host-defined and are not structured-output responses.

The Run registry configuration fingerprint records the effective non-secret Planner and Executor values (`base_url`, `model_name`, and Structured-output mode), rather than YAML structure or path; API keys are never logged, snapshotted, or fingerprinted. Consequently, a configuration change that changes either component's effective non-secret Provider configuration rejects resume, while API-key rotation and refactoring redundant inherited fields do not.

## Considered Options

- Read model configuration from environment variables, or let them override YAML — rejected because the CLI configuration must be explicit and reproducible.
- Require fully independent Planner and Executor blocks — rejected because common provider values should have one source while allowing distinct models or capabilities when needed.
- Keep JSON Schema as the only output mode — rejected because some OpenAI-compatible providers support JSON-object mode but not schema enforcement; local contract validation remains mandatory.
- Fingerprint the API key or complete YAML document — rejected because secrets must not enter durable records and equivalent effective configurations must resume consistently.
