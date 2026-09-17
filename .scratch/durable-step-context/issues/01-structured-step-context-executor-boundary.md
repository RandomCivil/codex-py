# 01: Structured Step context Executor boundary

**What to build:** An Executor invocation can receive readable Step context for its Tool-calling model and, after a successful Plan step, return an explicit outcome containing both the existing Step execution and a strictly validated Context update. The regular Agent continues recording only the Step execution, so its existing coordination behavior remains intact while callers gain the new handoff value.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] A valid forced completion receipt returns an ExecutionOutcome containing the completed Step execution and its Context update; invalid, unsafe, or malformed context makes the execution fail under the existing strict completion rules.
- [x] Every Executor request includes the supplied Step context as a separate fixed-format Markdown message, while retaining the existing JSON Model request and completion semantics.
- [x] Context paths accept only normalized relative POSIX values within the Agent working directory, and controlled Executor and Agent tests preserve the observable prior behavior.

## Answer

Implemented the structured Step context Executor boundary. `Executor.execute()` now accepts `StepContext` and returns `ExecutionOutcome`; successful strict completion receipts require `files_read`, `files_modified`, and `observations`, which are validated into an immutable `ContextUpdate`. Context is sent as a separate fixed-format Markdown message with all three sections present, including empty sections. The regular Agent records only `outcome.execution`, while DurableAgent passes its current graph context through the Executor boundary. Unsafe or malformed context preserves failed execution behavior.

Verification: `poetry run pytest -q` → 94 passed, 4 skipped.
