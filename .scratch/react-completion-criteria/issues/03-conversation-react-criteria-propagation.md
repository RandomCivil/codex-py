# 03: Conversation preserves ReAct completion-judge behavior

**What to build:** An interactive Conversation turn routed to ReAct receives the same ordered criteria and uses the same completion-judge behavior as a routed CLI execution, without changing Conversation persistence or recovery contracts.

**Blocked by:** 01: Task Analysis supplies ReAct completion criteria; 02: ReAct judges completion from settled tool evidence.

**Status:** resolved

- [x] Conversation mode selection retains the successful ReAct Task Analysis while keeping existing direct, tool-agent, Plan–execute, and analyzer-fallback behavior.
- [x] The Conversation runner factory passes criteria only to ReAct, allowing the existing ReAct completion judge to own progress, without changing persisted Conversation-turn or recovery contracts.
- [x] A controlled Conversation test verifies a routed ReAct turn receives ordered criteria and has the same criterion-enabled terminal behavior as CLI routing, without database or provider dependencies.

## Answer

- Conversation mode selection now retains the successful `TaskAnalysis` alongside the selected mode; analyzer failure still returns the existing `plan_execute` fallback without criteria.
- Ordered completion criteria are forwarded to ReAct runner construction only. Direct, tool-agent, and Plan–execute runner factories keep their existing argument contracts.
- Added controlled Conversation coverage for ordered criteria propagation, non-ReAct isolation, and a criterion-judged ReAct terminal answer without database or provider dependencies.
- Verification: `poetry run pytest -q` — 304 passed, 7 skipped.
