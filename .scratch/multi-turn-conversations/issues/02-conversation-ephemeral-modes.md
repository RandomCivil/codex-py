# 02: Conversation 的 Ephemeral modes

**What to build:** 让调用方在 Conversation turn 中显式选择既有 `tool_agent` 或 `react` Execution mode，并获得与 Direct turn 相同的历史输入、结果保留和后续可继续对话的体验。

**Blocked by:** 01: Conversation 基础流程（Direct mode）.

**Status:** ready-for-agent

- [x] `conversation run` 接受并保留 `tool_agent` 与 `react` 的显式 Execution mode；两个 mode 都获得完整公开 Conversation history 和独立当前输入。
- [x] completed 或 failed Ephemeral turn 的 answer/error 以公开 Conversation-turn result 写入历史，后续 turn 能看到该结果并可继续追加。
- [x] 中断或无法恢复的 Active Ephemeral turn 会安全终止为 failed，不会永久阻塞该 Conversation，且不重新执行未知的工具副作用。
- [x] 测试覆盖两个 mode 的显式选择、历史隔离、终态投影、失败后续聊和中断清理，不改变它们既有的工具权限、轮数或 Ephemeral 语义。

## Comments

- Implemented with TDD vertical slices in `tests/test_conversation_ephemeral.py`.
- Verification: `poetry run pytest -q` — 228 passed, 6 skipped.
