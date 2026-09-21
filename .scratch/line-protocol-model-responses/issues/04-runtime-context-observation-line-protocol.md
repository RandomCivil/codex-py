# 04: Runtime-context Observation Line Protocol slice

**What to build:** Runtime-context Observation generation and merging use validated `OBSERVATION` Line Protocol responses while retaining all asynchronous fallback and evidence-provenance behavior.

**Blocked by:** 01: Line Protocol core and Planner slice.

**Status:** ready-for-agent

- [ ] `OBSERVATION` explicitly represents rounds, source-round metadata, evidence categories, evidence text, and repeated tool-call IDs through declared fields and blocks.
- [ ] Observation and merge requests prompt for the protocol and omit provider-specific structured-output settings.
- [ ] Valid protocol output retains the existing confirmed-facts, reported-errors, model-inferences, and provenance semantics.
- [ ] Invalid protocol output follows the existing Observation failure/fallback policy and never becomes validated evidence.
- [ ] Tests preserve asynchronous Raw fallback behavior while proving valid decoding and malformed-output handling.
