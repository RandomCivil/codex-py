# 03: Plan–execute turn 与 Conversation recovery

**What to build:** 让 Conversation 能以 `plan_execute` 运行独立 durable Agent run，保留 blocked 结果，并由 `conversation resume` 把既有 Agent-run recovery 安全地回写到 Active Conversation turn。

**Blocked by:** 01: Conversation 基础流程（Direct mode）.

**Status:** ready-for-agent

- [x] `plan_execute` Conversation turn 创建并保留独立 durable Agent run，遵守现有 Checkpoint、配置快照、租约和恢复边界。
- [x] completed、failed 与 blocked Agent-run 结果准确投影为 Conversation-turn result；blocked 不降级为 failed，所有终态均成为下一轮的公开历史。
- [x] `conversation resume` 定位并恢复 Active durable turn，复用既有 Agent-run recovery 后原子回写 Conversation；活跃 durable turn 在恢复前拒绝追加。
- [x] 对已通过既有 `agent resume` 独立完成的关联 run，下一次 Conversation 操作会对账其终态后才决定追加或返回结果。
- [x] 测试覆盖 durable run 关联、blocked 投影、Conversation-owned recovery、独立 run recovery 对账，以及既有 `agent run`/`agent resume` 命令保持兼容。

## Comments

- Implemented with TDD vertical slices in `tests/test_conversation_plan_execute.py` and existing execution tests.
- Verification: `poetry run pytest -q` — 228 passed, 6 skipped.
