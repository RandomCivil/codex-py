# Codex Python Agent

一个基于 Python 3.12、LangGraph 和 OpenAI-compatible provider 的可恢复 Agent 运行时。CLI 的 `run` 会先分析目标，再从 `direct`、`tool_agent`、`react` 和 `plan_execute` 中确定性地选择执行模式；只有 `plan_execute` 会生成 Plan 并由 Executor 按顺序执行 Plan step。

`plan_execute` 的顶层协调状态和每个 Step execution 都使用 LangGraph checkpoint 持久化到 MySQL，因此进程重启后可以从最新 Checkpoint 恢复。其他三种模式是无 Checkpoint 的 ephemeral execution。MCP 工具由 Tool runtime 提供给 tool-enabled 模式；Plan 本身不包含具体工具调用。

## 特性

- 异步 LLM 适配器，支持 OpenAI Responses API 的文本流和事件流。
- 统一的四种 whole-task Execution mode：无工具的 `direct`、单轮工具调用的 `tool_agent`、迭代工具调用的 `react`，以及可恢复的 `plan_execute`。
- Task Analyzer + Task Router：`run` 只分析一次目标并按任务特征选择模式；`resume` 只恢复已有的 `plan_execute` run，不会重新路由。
- 基于 `model → ToolNode → model` 的 MCP 工具执行图。
- 同一模型响应中的 Tool-call batch 并发执行，等待全部调用完成后按请求顺序返回结果；单个调用失败不会丢弃同批的其他结果。
- 不可变、可校验的 Plan revision 和可审计的 Step execution 历史。
- MySQL checkpoint、运行登记和 60 秒独占 lease；lease 每 15 秒续租。
- 配置指纹校验，避免恢复时静默更换模型或 MCP 配置。
- Runtime-context component 异步生成和压缩 Tool-round Observation；最近三轮保留原始工具结果，Observation 失败时自动回退到原始证据。
- 通过 CLI 以一行 JSON 输出运行结果，便于脚本调用。
- 运行过程中将 LLM thought/output、Plan–Execute 状态以及 MCP tool call/result 实时输出到 stderr；最终结果仍只写到 stdout。
- `run` 先由 Task Router 描述性分析目标，再确定性地选择 direct、tool-agent、ReAct 或 Plan–execute。
- MySQL 持久化 Conversation：每轮保留独立 run、显式 Execution mode 与有序公开历史。

### 执行模式

| 模式 | 适用场景 | 工具调用 | 是否可恢复 |
| --- | --- | --- | --- |
| `direct` | 不需要外部环境或工具的回答 | 无 | 否 |
| `tool_agent` | 一次模型请求即可完成的简单工具任务 | 最多执行第一个 tool call | 否 |
| `react` | 需要根据工具结果反复判断的短期任务 | 多轮调用，直到结构化完成结果 | 否 |
| `plan_execute` | 长流程、多子目标、需要失败重规划的任务 | 每个 Plan step 由 Executor 执行 | 是 |

CLI 不接受手工 `--mode` 参数。Task Router 根据 Task Analyzer 的结构化分析选择模式；选择结果会在 `run` 或 Conversation turn 的 JSON 输出中返回。选定模式失败时不会自动升级到其他模式。

### 工具调用批次

模型在一次响应中发出的所有函数调用组成一个 Tool-call batch。Executor 会在同一个
Step execution 内并发启动这些调用，并等待整个批次 settle 后才再次调用模型或处理完成结果。
返回给模型的 ToolMessage 保留原始 tool-call ID 和请求顺序；单个工具失败会作为该调用的错误结果返回，
不会提前结束同批的其他调用。调用被取消时，未完成的工具会被取消，后续模型调用也不会继续。

Tool-call batch 只影响单个 Step 内的一轮工具调用，不会并行执行 Plan steps。同一批调用没有执行顺序保证；
存在依赖关系的操作应由模型拆分到后续响应中。

## 执行流程

一次 run 由 DurableAgent 按以下顺序协调：

1. 从 MySQL checkpoint 恢复状态；如果存在中断的 Step，先依据 `--recovery` 决定标记失败并重新规划，或直接进入 `blocked`。
2. Planner 生成初始 Plan；每个 Plan revision 最多包含三个版本。
3. 按 Plan 顺序逐个执行 Step。Step 开始前写入 `running` 状态，完成后保存 completion receipt 和上下文更新。
4. Step 失败时生成新的 Plan revision；第三个 revision 仍失败则进入 `blocked`。
5. 所有 Step 完成后进入 `completed`。每个关键节点通过 LangGraph checkpoint 持久化，进程重启后可以恢复。

```mermaid
flowchart TD
    S([START]) --> R{恢复状态?}
    R -->|blocked| B[blocked]
    R -->|interrupted| I[interrupted]
    R -->|正常| P[plan]
    P -->|还有 Step| M[mark_running]
    P -->|没有 Step| C[complete]
    M -->|Step 已失败| RP[replan]
    M -->|有待执行 Step| E[execute]
    M -->|全部完成| C
    E -->|成功且还有 Step| M
    E -->|失败| RP
    E -->|成功且全部完成| C
    RP -->|revision < 3| M
    RP -->|revision = 3| B
    C --> END([END])
    B --> END
    I --> END
```

### 单个 Step 的 Executor 图

Executor 为一个 Step 构建并运行独立的 LangGraph。`ToolNode` 会处理当前模型响应中的整个
Tool-call batch；没有工具调用时直接结束工具图。工具图结束后，Executor 再单独请求严格 JSON
格式的 completion receipt，用于判断 completion criterion 并生成后续 Step 的上下文。

```mermaid
flowchart LR
    S([START]) --> M[mark_running]
    M --> L[model]
    L -->|有 tool_calls 且未超出轮数| T[tools / ToolNode]
    T -->|等待整个 Tool-call batch| L
    L -->|无 tool_calls 或达到轮数上限| E([END])
    E --> Q[completion receipt]
    Q --> X{criterion met?}
    X -->|是| D[Step completed]
    X -->|否或格式无效| F[Step failed]
```

Plan steps 始终串行；只有同一模型响应内相互独立的工具调用会在 `ToolNode` 中并发执行。

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

运行前设置 MySQL 连接环境变量：

```bash
export CODEX_MYSQL_URL='mysql://user:password@127.0.0.1:3306/codex'
```

`CODEX_MYSQL_URL` 只用于 MySQL 连接，不能省略。凭据不会写入运行配置快照；Checkpoint 状态目前按原样保存，请为该数据库设置合适的访问控制。

`run`、`resume` 和 Conversation 命令都需要显式的 `--config` YAML；`migrate` 只需要 `CODEX_MYSQL_URL`，不需要模型配置。配置加载时，`model` 提供共享的 Component provider configuration，组件节点只覆盖需要变化的字段：

```text
model
├── planner              ─┐
├── executor              ├─ 继承共享配置，可单独覆盖
├── task_analyzer        ─┘  默认继承 planner
├── runtime_context      （可选，默认使用所属工具模式的模型）
├── direct
├── tool_agent
└── react
```

Planner 和 Executor 的 provider 配置必须写在显式传入的 YAML 文件中。文件使用共享的 `model` 默认值，并可用 `planner`、`executor` 分别覆盖任意字段：

```yaml
# agent.yaml
model:
  base_url: https://your-provider.example/v1
  api_key: your-api-key
  model_name: gpt-4o-mini
  response_format: json_schema

planner:
  model_name: planner-model

executor:
  response_format: json_object

# Whole-task execution mode overrides are optional and inherit from model.
direct:
  model_name: direct-model

tool_agent:
  base_url: https://tools-provider.example/v1

react:
  response_format: json_schema

# Optional; when omitted, Observation calls use the owning ReAct/Executor model.
# When present, it inherits model and generates and compacts tool-round Observations.
runtime_context:
  model_name: context-summary-model
  response_format: json_object

# Optional; inherits the effective planner configuration.
task_analyzer:
  model_name: task-analysis-model
```

`response_format` 只能是 `json_schema` 或 `json_object`。需要结构化结果的请求会使用所属组件的有效格式；`json_schema` 请求使用该调用的严格 schema，`json_object` 请求只要求返回 JSON object，Agent 仍会在本地严格校验结果。Executor 的工具选择请求不绑定结构化输出，以保留 MCP function call 能力；最终 completion receipt 使用 Executor 的有效格式。
`direct`、`tool_agent` 和 `react` 使用各自 provider 的有效 `response_format`（未显式配置时继承共享模型配置，最终回退为 `json_schema`）。CLI 不提供 mode selector；单次 `run` 与每个 Conversation turn 都由 Task Router 通过 Python factory 组合并选择模式。
`runtime_context` 使用 Structured-output mode，默认继承共享模型配置；它只负责生成与合并 Tool-round Observation，可单独指定 provider、模型和 `response_format`。`task_analyzer` 使用 Structured-output mode，并默认继承 Planner 的有效配置；它只描述任务特征，不能选择 Execution mode。`run` 始终先调用一次 Task Router：无工具目标使用 direct，短且确定的工具目标使用 tool-agent，长周期/多子目标/高重规划目标使用 Plan–execute，其余工具目标使用 ReAct。CLI 结果会包含 `execution_mode`、`execution`、`analysis` 和（分析失败时）安全的 `analysis_error`。`resume` 仅恢复既有的 Plan–execute Agent run，不会重新分析或更换模式。

YAML 中的 `api_key` 是明文配置。请限制配置文件的文件权限，例如：

```bash
chmod 600 agent.yaml
```

不要把 Planner 或 Executor 的 model 配置放入环境变量；CLI 不会从 `OPENAI_*` 环境变量读取或覆盖 YAML。配置文件路径也不会写入运行快照，API key 不会写入快照或 fingerprint。

Executor 默认连接：

```text
command: poetry
args:    run atom-mcp
cwd:     /home/xzp/workspace/atom-mcp
```

如需使用其他 MCP host，请在集成 `Executor` 时传入自定义 stdio 配置。

## CLI

所有命令都把最终结果以一行 JSON 写到 stdout，运行日志写到 stderr，因此可以安全地被脚本消费：

```bash
poetry run agent <command> [options]
```

首次使用时显式初始化数据库 schema：

```bash
poetry run agent migrate
```

开始一个新的 Agent run：

```bash
poetry run agent run --goal '检查项目中的待办事项并整理摘要' --config agent.yaml --cwd /path/to/project
```

`run` 的 JSON 结果中的 `status` 是所选 Execution mode 的 `completed`、`failed` 或 `blocked` 状态。

一个不需要工具的目标可能返回 `direct`，简单工具任务可能返回 `tool_agent`，需要迭代观察时返回 `react`，长流程则返回 `plan_execute`。完整结果还可能包含 `execution`、`analysis` 和分析失败时的安全 `analysis_error`：

```json
{
  "command": "run",
  "run_id": "550e8400-e29b-41d4-a716-446655440000",
  "execution_mode": "react",
  "status": "completed",
  "execution": {"answer": "…", "status": "completed", "error": null},
  "analysis": {"task_type": "retrieval", "needs_tools": true}
}
```

降低日志量时使用 `--log-level error`。该选项不会改变执行结果，只隐藏 LLM 请求、响应、流式事件和中间 thought/output 的详细内容。

`--cwd` 指定 Agent 操作的项目目录。它会被强制写入每一个 Atom MCP 工具调用的 `cwd` 参数；未指定时使用启动 `agent` 命令时的当前目录。恢复 run 时应使用与原 run 相同的 `--cwd`，该目录属于持久化配置的一部分。

命令会输出 UUIDv4 `run_id`，例如：

```json
{"command":"run","run_id":"550e8400-e29b-41d4-a716-446655440000","status":"completed"}
```

恢复未完成的 run：

```bash
poetry run agent resume --run-id <uuidv4> --config agent.yaml
```

如果上次中断时存在正在执行的 Step，必须显式选择恢复策略：

```bash
poetry run agent resume --run-id <uuidv4> --config agent.yaml --recovery fail
poetry run agent resume --run-id <uuidv4> --config agent.yaml --recovery abort
```

当前 Atom MCP 工具没有声明幂等性，因此 `retry` 会被拒绝。`fail` 将该 Step 标记为失败并触发 replanning；`abort` 将 run 标记为 `blocked`。已完成或已阻塞的 run 再次 resume 时直接返回保存的终态结果，不会重新调用模型或工具。

### Conversation

Conversation 是以 `conv_id` 标识的持久化多轮交互。每个 turn 都有独立的 UUIDv4 `run_id`；Task Analyzer 会选择并记录既有的 `direct`、`tool_agent`、`react` 或 `plan_execute` mode。它不会改变既有单次 `run` / `resume` 的语义。

新建 Conversation（省略 `--conv-id`）并运行首轮：

```bash
poetry run agent conversation run --input '整理当前项目的待办事项' --config agent.yaml --cwd /path/to/project
```

结果会返回本次 turn，而不是完整历史：

```json
{"command":"conversation","conv_id":"550e8400-e29b-41d4-a716-446655440000","sequence":1,"run_id":"550e8400-e29b-41d4-a716-446655440001","execution_mode":"direct","status":"completed","answer":"…","error":null}
```

使用返回的 ID 追加下一轮；每轮都会由 Task Analyzer 重新选择 mode：

```bash
poetry run agent conversation run --conv-id <uuidv4> --input '根据上面的结果列出下一步' --config agent.yaml --cwd /path/to/project
```

读取完整、有序的公开历史：

```bash
poetry run agent conversation show --conv-id <uuidv4>
```

同一 Conversation 只允许一个 `pending` 或 `running` turn。并发追加会以退出码 2 返回 `conversation busy`，并包含活跃 turn 的 `sequence` 和 `run_id`；不会排队或并行执行。终态（`completed`、`failed`、`blocked`）后的 Conversation 可以继续追加。

每次模型调用都会收到此前所有 turn 的公开结构化 history（序号、用户输入、mode、run ID、状态与 answer/error）以及独立的当前输入；Checkpoint、Plan、Step execution、原始 tool result 和凭据不会进入 history。当前版本不截断或压缩 history；provider 因上下文过大而拒绝请求时，该 turn 会记录为失败。

仅 `plan_execute` turn 可以恢复。它会复用关联 Agent run 的 recovery 和原始有效 tool working directory：

```bash
poetry run agent conversation resume --conv-id <uuidv4> --config agent.yaml
poetry run agent conversation resume --conv-id <uuidv4> --config agent.yaml --recovery fail
poetry run agent conversation resume --conv-id <uuidv4> --config agent.yaml --recovery abort
```

中断的 ephemeral turn（`direct`、`tool_agent`、`react`）不可恢复；下一次 Conversation 操作会先将它安全标记为失败。独立使用 `agent resume` 完成关联 durable run 后，下一次 Conversation 操作会先对账该终态再决定是否追加。

CLI 输出包含 `command`、`status`、`run_id` 等字段；发生错误时还包含 `error`。退出码如下：

运行过程日志使用带有 `[llm thought]`、`[llm output]`、`[plan]`、`[execute]`、`[tool call]` 和 `[tool result]` 前缀的文本格式写入 stderr，因此不会污染 stdout 中的一行 JSON 结果。

每行运行日志都会以 `run_id` 作为前缀。默认日志级别为 `info`。`[llm context]` 请求上下文与 `[llm final]` 最终返回仅在 `info` 级别输出；每次 LLM 调用完成都会在所有级别输出 `[llm usage]` token usage。需要降低输出量时可使用 `--log-level error`：隐藏 LLM 的请求/响应详情、流式事件与中间 thought/output，只保留 usage、Plan/Execute 状态和工具调用信息；tool result 只显示工具名，tool call 保留传入参数。

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
