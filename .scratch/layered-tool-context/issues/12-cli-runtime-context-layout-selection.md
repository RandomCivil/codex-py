# 12: CLI Runtime-context layout selection

**What to build:** Expose the selected Runtime-context layout through the CLI
and retain it as durable-run configuration identity. A caller can opt into
`messages` across run, resume, conversation, and chat without changing existing
default `grouped` behavior.

**Blocked by:** 10: ReAct messages layout; 11: Plan-step messages layout.

**Status:** ready-for-agent

- [ ] Every supported CLI entry accepts `--runtime-context-layout grouped|messages`
  and defaults to `grouped`.
- [ ] The selected layout reaches applicable ReAct and Plan-step execution;
  passing it to a mode without Runtime context is accepted and has no effect.
- [ ] A durable run records the layout as configuration identity, and resume
  accepts only the value matching the created run; omission compares as
  `grouped`.
- [ ] Conversation and chat propagate the selected layout for newly created
  turns and enforce the existing recovery configuration contract.
- [ ] Existing invocations without the flag preserve their current output and
  validation behavior.

See [the feature specification](../spec.md) for the full contract.
