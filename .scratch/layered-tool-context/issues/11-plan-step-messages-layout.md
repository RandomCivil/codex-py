# 11: Plan-step messages layout

**What to build:** Let the Plan-step Executor use the `messages` Runtime-context
layout for operational and Completion-judge requests while preserving the
Plan-step evidence boundary. Successful Tool evidence reaches the judge in the
new native message form; failed Tool evidence remains available to the
operational correction loop.

**Blocked by:** 09: Runtime-context messages rendering and budget.

**Status:** ready-for-agent

- [ ] Plan-step operational requests use the selected `messages` ordering with
  their fixed system instructions, durable step state, Goal, and current
  acceptance criterion.
- [ ] The Completion judge uses the selected layout but retains its existing
  evidence-only selection and reconstructs only successful Raw Tool pairs.
- [ ] Failed Tool evidence remains in the operational correction request and
  does not become completion evidence.
- [ ] The existing `grouped` Plan-step request remains unchanged when selected.

See [the feature specification](../spec.md) for the full contract.
