# 06: Remove structured-output configuration and finalize migration

**What to build:** The application has Line Protocol as its only non-tool model-response contract: provider-specific JSON modes and configuration are removed, stale YAML fails clearly, and operational behavior/documentation reflects the final contract.

**Blocked by:** 01: Line Protocol core and Planner slice; 02: Task-analysis Line Protocol slice; 03: Ephemeral answer-mode Line Protocol slice; 04: Runtime-context Observation Line Protocol slice; 05: Tool-loop completion Line Protocol slice.

**Status:** ready-for-agent

- [ ] All former JSON Schema/JSON-object response-mode APIs, provider request settings, prompts, and tests are removed or migrated; no non-tool model path retains provider output formatting.
- [ ] Component provider configuration and inheritance omit `response_format`; a legacy field fails before model or Tool work with a migration-focused configuration error.
- [ ] Non-secret configuration snapshots and resume fingerprints omit output mode while preserving existing endpoint/model drift and API-key rotation behavior.
- [ ] Documentation describes Line Protocol as the sole contract and no longer instructs operators to select JSON response modes.
- [ ] Full regression coverage proves every registered response type, the removal/migration behavior, and unaffected native-tool, recovery, and durable execution boundaries.
