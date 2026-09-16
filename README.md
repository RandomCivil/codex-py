# Codex Python Agent

一个基于 Python 3.12、LangGraph 和 OpenAI-compatible provider 的可恢复 Agent。Agent 将用户目标交给 Planner 生成完整的、严格 JSON 校验的 Plan，再由 Executor 按顺序执行每个 Plan step。失败时最多生成三个不可变 Plan revision；预算耗尽后返回 `blocked`。

Agent 的顶层协调状态和每个 Step execution 都使用 LangGraph checkpoint 持久化到 MySQL，因此进程重启后可以从最新 Checkpoint 恢复。MCP 工具由 Executor 在运行时选择和调用，Plan 本身不包含具体工具调用。

## 特性

- 异步 LLM 适配器，支持 OpenAI Responses API 的文本流和事件流。
- 基于 `model → ToolNode → model` 的 MCP 工具执行图。
- 不可变、可校验的 Plan revision 和可审计的 Step execution 历史。
- MySQL checkpoint、运行登记和 60 秒独占 lease；lease 每 15 秒续租。
- 配置指纹校验，避免恢复时静默更换模型或 MCP 配置。
- 通过 CLI 以一行 JSON 输出运行结果，便于脚本调用。
- 运行过程中将 LLM thought/output、Plan–Execute 状态以及 MCP tool call/result 实时输出到 stderr；最终结果仍只写到 stdout。

## 安装

要求：

- Python 3.12+
- Poetry
- MySQL 8.0.19–9.5
- 可访问的 OpenAI-compatible Responses API
- 可执行的 Atom MCP 服务（默认通过 `poetry run atom-mcp` 启动）

```bash
poetry install
```

## 配置

运行前设置以下环境变量：

```bash
export CODEX_MYSQL_URL='mysql://user:password@127.0.0.1:3306/codex'
export OPENAI_BASE_URL='https://your-provider.example/v1'
export OPENAI_API_KEY='your-api-key'
export OPENAI_MODEL='gpt-4o-mini'       # 可选，默认 gpt-4o-mini
```

`CODEX_MYSQL_URL` 只用于 MySQL 连接，不能省略。凭据不会写入运行配置快照；Checkpoint 状态目前按原样保存，请为该数据库设置合适的访问控制。

Executor 默认连接：

```text
command: poetry
args:    run atom-mcp
cwd:     /home/xzp/workspace/atom-mcp
```

如需使用其他 MCP host，请在集成 `Executor` 时传入自定义 stdio 配置。

## CLI

首次使用时显式初始化数据库 schema：

```bash
poetry run agent migrate
```

开始一个新的 Agent run：

```bash
poetry run agent run --goal '检查项目中的待办事项并整理摘要'
```

命令会输出 UUIDv4 `run_id`，例如：

```json
{"command":"run","run_id":"550e8400-e29b-41d4-a716-446655440000","status":"completed"}
```

恢复未完成的 run：

```bash
poetry run agent resume --run-id <uuidv4>
```

如果上次中断时存在正在执行的 Step，必须显式选择恢复策略：

```bash
poetry run agent resume --run-id <uuidv4> --recovery fail
poetry run agent resume --run-id <uuidv4> --recovery abort
```

当前 Atom MCP 工具没有声明幂等性，因此 `retry` 会被拒绝。`fail` 将该 Step 标记为失败并触发 replanning；`abort` 将 run 标记为 `blocked`。已完成或已阻塞的 run 再次 resume 时直接返回保存的终态结果，不会重新调用模型或工具。

CLI 输出包含 `command`、`status`、`run_id` 等字段；发生错误时还包含 `error`。退出码如下：

运行过程日志使用带有 `[llm thought]`、`[llm output]`、`[plan]`、`[execute]`、`[tool call]` 和 `[tool result]` 前缀的文本格式写入 stderr，因此不会污染 stdout 中的一行 JSON 结果。

| 退出码 | 含义 |
| ---: | --- |
| 0 | completed |
| 1 | blocked |
| 2 | run busy，另一个进程持有 lease |
| 3 | configuration 错误 |
| 4 | persistence 错误 |
| 5 | CLI 参数或 recovery decision 无效 |

## 开发与测试

运行单元测试：

```bash
poetry run pytest
```

MySQL 集成测试不会使用运行时数据库配置。为测试准备专用 schema，并设置：

```bash
export CODEX_TEST_MYSQL_URL='mysql://user:password@127.0.0.1:3306/codex_test'
poetry run pytest -m integration
```

未设置 `CODEX_TEST_MYSQL_URL` 时，集成测试会带原因跳过。

## 项目结构

```text
agent/     Agent 协调、Planner、Executor、持久化和 CLI
llm/       OpenAI-compatible Responses API 异步适配器
memory/    AgentState、Plan 和 StepExecution 领域模型
tests/     单元测试和 MySQL 集成测试
docs/adr/  关键架构决策
```

更多术语和边界约定见 [CONTEXT.md](CONTEXT.md)，架构决策见 [docs/adr](docs/adr/)。
