# 02: Durable Step context handoff

**What to build:** A Durable Agent run accumulates the Context updates from successful Plan steps and hands the resulting Step context to each later Plan step, including when the Agent run resumes from its latest Checkpoint. File paths remain stable and de-duplicated, while observations retain their actual Plan-step attribution.

**Blocked by:** 01 — Structured Step context Executor boundary.

**Status:** ready-for-agent

- [ ] A successful Step execution merges its Context update into checkpointed Graph state; duplicate file paths keep first-seen order and each observation is attributed by the system to its revision and Plan-step ID.
- [ ] A failed, interrupted, or malformed Step execution makes no Step-context change, and subsequent or replanned steps receive only prior successful context.
- [ ] Controlled Durable Agent tests demonstrate cross-step handoff and checkpoint-resume preservation without changing serialized Agent-state or terminal-result contracts.
