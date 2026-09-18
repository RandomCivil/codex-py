# 05: Mode-specific YAML provider configuration

**What to build:** An operator can declare provider defaults and per-mode overrides for all Execution modes, then programmatically compose compatible mode runners without changing existing CLI invocation behavior.

**Blocked by:** 02: Shared Tool runtime and tool-agent mode; 03: Bounded ReAct mode; 04: Plan–execute mode adapter.

**Status:** resolved

- [x] YAML accepts optional `direct`, `tool_agent`, and `react` overrides alongside existing component overrides, with every component inheriting effective values from shared `model` defaults.
- [x] Direct and tool-agent validate endpoint, credential, and model configuration; ReAct additionally validates its Structured-output mode and uses it for its final goal-satisfaction response.
- [x] Existing Planner/Plan-step Executor configuration, non-secret snapshot/fingerprint rules, and CLI command behavior remain compatible; no CLI mode selector is added.
- [x] Configuration-boundary and public composition tests cover valid inheritance, independent overrides, invalid documents, ReAct output-mode validation, and unchanged legacy configuration behavior.

## Comments

- Added YAML provider overrides for `direct`, `tool_agent`, and `react`, with shared-default inheritance and mode-specific validation.
- Extended the execution factory to compose configured provider models and apply ReAct structured output to its goal-completion response.
- Updated operator documentation and ADR-0006; CLI and durable plan-execute configuration snapshots remain Planner/Executor-only.
- Verification: `poetry run pytest -q` — 148 passed, 5 skipped.
