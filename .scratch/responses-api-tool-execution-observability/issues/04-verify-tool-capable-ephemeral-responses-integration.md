# 04: Verify the tool-capable Ephemeral Responses integration

**What to build:** The configured execution-mode factory demonstrably gives both Tool-agent mode and ReAct mode the shared Responses API path, preserving their result contracts and emitting complete request-level observability across the full mode lifecycle.

**Blocked by:** 02: Run Tool-agent mode through the Responses API; 03: Run ReAct mode through the Responses API.

**Status:** ready-for-agent

- [ ] Controlled end-to-end execution-mode tests verify the configured factory selects the Responses API path for both modes without live provider credentials.
- [ ] The tests verify each nested provider request has independently attributable timing and usage output while user-visible final results remain unchanged.
- [ ] The supported provider configuration and existing mode-selection behavior remain compatible with the migrated modes.
