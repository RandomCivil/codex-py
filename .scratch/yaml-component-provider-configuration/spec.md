# YAML component Provider configuration

Status: ready-for-agent

## Problem Statement

As an operator, I cannot configure the Planner and Executor independently from one explicit, durable runtime configuration. The CLI currently relies on a shared environment-derived model selection, which makes per-component provider credentials, endpoints, models, and Structured-output modes unavailable and makes a resumed Agent run harder to reproduce.

## Solution

Require `run` and `resume` to receive an explicit YAML configuration path. The YAML file defines a shared Component provider configuration and optional Planner and Executor overrides. It supports `json_schema` and `json_object` Structured-output modes, preserves local validation of Plans and Execution results, and fingerprints each component's effective non-secret configuration so resumes reject meaningful configuration drift without storing API keys.

## User Stories

1. As an operator, I want `run` to require an explicit YAML configuration path, so that every Agent run has an intentional and auditable model configuration source.
2. As an operator, I want `resume` to require the same kind of configuration path, so that a resumed Agent run cannot silently choose a different provider or model.
3. As an operator, I want one shared Provider configuration, so that common endpoint, credential, model, and output-mode values are declared once.
4. As an operator, I want to override individual shared values for the Planner, so that planning can use a model suited to whole-goal Plan generation.
5. As an operator, I want to override individual shared values for the Executor, so that Step execution and tool selection can use a different compatible model when needed.
6. As an operator, I want each component to inherit omitted values from the shared configuration, so that overrides remain concise without creating partial runtime configurations.
7. As an operator, I want a configuration to fail before model or Tool execution when it is missing, malformed, incomplete, or contains an unknown field, so that a typo cannot cause an unintended model invocation.
8. As an operator, I want configuration failures to use the CLI's existing configuration result and exit code, so that automation can distinguish them from invalid invocation and persistence failures.
9. As an operator, I want to select `json_schema` for a component, so that a compatible provider can enforce the supplied structured-response schema.
10. As an operator, I want to select `json_object` for a component, so that a provider lacking schema enforcement can still produce a JSON object for local contract validation.
11. As an operator, I want malformed or contract-incompatible `json_object` output rejected locally, so that choosing provider compatibility never weakens Plan or Execution result invariants.
12. As an operator, I want the Planner's configured Structured-output mode to affect its Plan request, so that the complete Plan remains machine-validatable with either provider mode.
13. As an operator, I want the Executor's configured Structured-output mode to affect its final completion receipt, so that the Step execution handoff stays validated with either provider mode.
14. As an operator, I want every Executor model request, including tool-selection requests, to use the Executor's effective Provider configuration, so that its endpoint, credential, and model cannot diverge from its completion processing.
15. As an operator, I want MCP tool calls to remain host-defined rather than structured-output responses, so that tool calling retains the boundary established for the Executor.
16. As an operator, I want API keys to be written directly in the YAML file and never sourced from model-related environment variables, so that configuration has exactly one model-configuration authority.
17. As an operator, I want API keys excluded from output, persisted configuration snapshots, and configuration fingerprints, so that credentials are not exposed by diagnostics or durable metadata.
18. As an operator, I want to rotate an API key without invalidating a resumable Agent run, so that credential maintenance does not discard durable progress.
19. As an operator, I want changing an effective component endpoint, model, or Structured-output mode to reject resume, so that the Agent run stays reproducible across interruption.
20. As an operator, I want semantically equivalent YAML inheritance and an unchanged effective configuration to resume successfully, so that harmless configuration refactoring does not block recovery.
21. As a Python integrator, I want existing direct Planner and Executor construction seams to remain usable, so that embedding the Agent does not require YAML file I/O.
22. As an operator, I want database migration to remain independent of the YAML file, so that schema initialization needs only its existing database configuration.
23. As a maintainer, I want the documented CLI configuration workflow updated when the feature ships, so that operators are not instructed to configure models through obsolete environment variables.

## Implementation Decisions

- Add a YAML parsing dependency and a dedicated configuration boundary that validates the document before runtime construction. YAML is the sole source for Planner and Executor Provider configuration; no model-related environment-variable fallback or interpolation is supported.
- The document has exactly three allowed top-level mappings: `model`, `planner`, and `executor`. The shared `model` mapping is the inheritance base. `planner` and `executor` are optional override mappings; each can override any subset of `base_url`, `api_key`, `model_name`, and `response_format`.
- The effective Component provider configuration for both Planner and Executor must contain non-empty string `base_url`, `api_key`, and `model_name`, plus `response_format` set to exactly `json_schema` or `json_object`. Unknown keys, invalid YAML, invalid field types, empty required values, and absent effective values are configuration errors.
- Add `--config` to `run` and `resume` and require it for both commands. Do not auto-discover a conventional configuration filename. `migrate` accepts no model configuration because it only initializes database schema.
- Construct the Planner using its effective Component provider configuration. Construct the Executor with its own effective Component provider configuration for tool-selection and final-completion model requests.
- `response_format` controls only the Planner's structured Plan request and Executor's final structured completion receipt. Executor tool-selection requests must still be capable of emitting MCP function calls and MCP tool schemas remain host-defined.
- In `json_schema` mode, submit the existing strict schemas to the provider. In `json_object` mode, request a JSON object and retain the existing local parsing and invariant validation for Plans and completion receipts.
- Refactor the Run registry's non-secret configuration snapshot to represent the effective Planner and Executor `base_url`, `model_name`, and Structured-output mode, alongside existing MCP execution configuration. It must exclude API keys and configuration-file paths. The configuration fingerprint derives from this normalized snapshot.
- Resume must compare the normalized effective configuration before model or Tool execution. API-key changes and redundant inheritance edits do not change the fingerprint; changing an effective endpoint, model name, or Structured-output mode does.
- Preserve direct injection seams for Planner and Executor used by Python integration and unit tests; YAML parsing belongs at the CLI/runtime composition boundary rather than in these domain components.
- Update operator documentation with the explicit YAML workflow, a shared-default/override example, API-key file-protection guidance, and the precise `json_schema` versus `json_object` behavior.

## Testing Decisions

- Test external behavior rather than internal parsing mechanics: a valid YAML document must result in the expected effective Planner and Executor Provider configurations; invalid documents must produce the stable configuration CLI result before any model or Tool execution.
- Use the existing CLI tests as the highest test seam for required `--config`, `migrate` independence, configuration-error exit semantics, and forwarding the resolved configuration into runtime composition.
- Add focused configuration-boundary tests for inheritance, independent per-field overrides, required values, allowed values, invalid YAML, and rejected unknown fields.
- Extend existing Planner, Executor, and LLM request tests to prove that `json_schema` preserves strict provider request shapes, `json_object` uses the compatible provider request shape, and both paths continue to enforce local Plan and completion-receipt validation.
- Extend existing Run registry tests to prove that credentials and configuration-file paths are absent from snapshots and fingerprints; API-key rotation and equivalent inherited values resume, while effective Planner or Executor endpoint/model/output-mode drift is rejected before work begins.
- Add runtime composition tests proving both components receive their own effective Provider configuration, including the Executor's tool-selection model path, without requiring a live provider or MCP host.
- Preserve existing integration-test isolation: no new test may use runtime credentials or database URLs as a fallback.

## Out of Scope

- Model configuration from environment variables, environment-variable interpolation inside YAML, secret-manager integration, encrypted configuration files, and automatic configuration-file discovery.
- Configuring MCP host transport, tool allowlists, or MCP tool schemas through this YAML document.
- Changing Plan semantics, Plan-revision limits, Executor tool authority, Tool-call batch behavior, or the durable recovery protocol.
- Making direct Python callers use YAML rather than their existing injection interfaces.
- Database migration behavior beyond retaining its existing independence from model configuration.

## Further Notes

The YAML file deliberately contains plaintext API keys. Operators are responsible for restricting its filesystem permissions. Checkpoint state remains subject to the existing database access-control guidance. This specification follows ADR-0001's Plan/Executor boundary, ADR-0002's host-defined MCP tool boundary, ADR-0003's non-secret resume compatibility rule, and ADR-0006's accepted YAML configuration decision.
