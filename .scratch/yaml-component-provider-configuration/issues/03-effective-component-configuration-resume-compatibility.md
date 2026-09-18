# 03: Effective Component provider configuration resume compatibility

**What to build:** An operator can safely resume only with equivalent effective non-secret Planner and Executor Provider configurations. API-key rotation and equivalent YAML inheritance retain resumability, while a changed effective endpoint, model name, or Structured-output mode is rejected before model or Tool execution.

**Blocked by:** 01: Configuration file entry and validation.

**Status:** ready-for-agent

- [ ] The Run registry snapshots and fingerprints effective non-secret component values without recording API keys or configuration-file paths.
- [ ] Resume permits API-key rotation and semantically equivalent inheritance.
- [ ] Resume rejects effective component configuration drift before any model or Tool execution.

