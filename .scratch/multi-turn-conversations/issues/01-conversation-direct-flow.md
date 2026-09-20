# 01: Conversation 基础流程（Direct mode）

**What to build:** 让 CLI 调用方能不传 `conv_id` 新建 Conversation，或传入已有 ID 追加一个 Direct-mode Conversation turn；每轮将此前完整结构化 Conversation history 与独立当前输入交给执行，并能通过 `conversation show` 读取有序历史。

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [x] Conversation 和 Conversation turn 持久化到 MySQL；新会话与首轮原子创建、每轮独立 run ID、严格递增 turn 序号，以及一会话仅一个 Active Conversation turn 均可观察地成立。
- [x] `conversation run` 与 `conversation show` 提供稳定 JSON：省略 `conv_id` 返回新 UUIDv4，已有 ID 追加，未知/无效 ID 被拒绝；正常运行结果仅含当前 turn，show 返回完整历史。
- [x] Direct turn 的模型输入包含此前公开结构化历史与独立当前输入，不泄漏 Checkpoint、Plan、Step execution 或原始工具输出。
- [x] 并发追加不会并行执行；冲突调用稳定返回 `conversation busy` 和安全的活跃工作标识。
- [x] Conversation 应用服务的行为测试、薄 CLI 契约测试和 MySQL 迁移/条件追加集成测试覆盖上述结果。
