# 06: Tighten Plan-step Executor fixed prompts by Structured-output mode

**What to build:** The Plan-step Executor minimizes fixed operational and completion-receipt prompt text whenever its strict Structured-output contract already enforces it, while retaining completion-criterion evidence, recovery reconciliation, and JSON-object compatibility.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Strict-schema Plan-step completion requests omit only receipt-shape instructions guaranteed by the configured contract; JSON-object requests retain the textual field and shape constraints needed for local receipt validation.
- [ ] Operational tool-loop guidance, Step context, Runtime context, recovery safeguards, completion criteria, Tool execution, completion receipts, and durable Context updates retain their existing behavior.
