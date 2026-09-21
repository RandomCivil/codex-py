# YAML component Provider configuration

**Status: superseded by ADR-0017.** CLI `run` and `resume` require an explicit `--config` path to a YAML file; environment variables are not a model-configuration source and `migrate` does not read this file. The document supplies a shared `model` default and optional component overrides for `base_url`, plaintext `api_key`, and `model_name`. Invalid YAML, unknown fields, invalid values, or missing effective fields are configuration errors before any model or tool work.

All non-tool model responses use the locally validated Line Protocol defined by ADR-0017. Native MCP tool calls remain provider-native.

The Run registry configuration fingerprint records the effective non-secret component values (`base_url` and `model_name`), rather than YAML structure or path; API keys are never logged, snapshotted, or fingerprinted. Consequently, a configuration change that changes either component's effective non-secret Provider configuration rejects resume, while API-key rotation and refactoring redundant inherited fields do not.

## Considered Options

- Read model configuration from environment variables, or let them override YAML — rejected because the CLI configuration must be explicit and reproducible.
- Require fully independent Planner and Executor blocks — rejected because common provider values should have one source while allowing distinct models or capabilities when needed.
- Fingerprint the API key or complete YAML document — rejected because secrets must not enter durable records and equivalent effective configurations must resume consistently.
