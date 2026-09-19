# 04: Tighten Planner and Task Analyzer static instructions

**What to build:** Operators pay fewer input tokens for Planner and Task Analyzer requests whose static prompts miss a provider cache, while Plan generation and deterministic Execution-mode routing retain their current validated contracts.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] For both components, strict-schema requests remove only instruction text duplicated by the schema; JSON-object requests retain textual JSON, field, and shape constraints needed for compatibility and local validation.
- [ ] Public Text stream requests continue to keep fixed instructions separate from goals and Agent state; valid, malformed, and unsupported Planner or Task Analyzer output retain their current validation and routing behavior.
