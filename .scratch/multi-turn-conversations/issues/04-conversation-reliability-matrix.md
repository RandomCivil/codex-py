# 04: 完整模式矩阵与 MySQL 可靠性验证

**What to build:** 验证所有既有 Execution modes 都能作为严格有序、可恢复且不泄漏内部状态的 Conversation turn 使用，并确认 MySQL 持久化在跨进程竞争和重启后维持 Conversation 契约。

**Blocked by:** 02: Conversation 的 Ephemeral modes; 03: Plan–execute turn 与 Conversation recovery.

**Status:** ready-for-agent

- [x] 从 CLI 到 Conversation 应用服务验证四个 mode 的新建、追加、展示、历史输入和当前 turn 响应契约。
- [x] MySQL 集成测试验证迁移幂等、跨进程条件追加、一会话一个 Active turn、终态后续追加，以及恢复后的顺序保持；测试仅使用专用测试数据库。
- [x] 验证所有历史只含公开 turn 字段，配置/凭据边界和既有单次 Agent 命令行为不回归。
- [x] 运行并通过相关单元、CLI 与可用 MySQL 集成测试，记录因未配置专用测试数据库而跳过的集成测试。

## Comments

- Completed the reliability matrix with explicit CLI coverage for all four Execution modes, stable current-turn resume output, and public-history boundaries.
- Fixed restart reconciliation so interrupted ephemeral turns are terminally failed without querying the durable Agent-run registry.
- Added idempotent migration and competing-append MySQL integration coverage using `CODEX_TEST_MYSQL_URL` only.
- Verification: `poetry run pytest -q` — 234 passed, 7 skipped; MySQL integration tests were skipped because `CODEX_TEST_MYSQL_URL` is not configured.
