# Codex Python

This context coordinates language-model interactions for the Codex Python agent.

## Language

**OpenAI-compatible provider**:
A service exposing the OpenAI Responses API through a caller-supplied base URL, API key, and model name.
_Avoid_: OpenAI provider, model provider

**Structured-output mode**:
The OpenAI-compatible provider request mode used for a component's final structured response: `json_schema`, which asks the provider to enforce a supplied schema, or `json_object`, which asks it only to return a JSON object and relies on local validation for the component's contract.
_Avoid_: JSON mode, response type

**Component provider configuration**:
The effective OpenAI-compatible provider credentials, endpoint, model name, and, where structured output is required, Structured-output mode used by a named language-model component. Components may inherit shared values, then apply component-specific overrides.
_Avoid_: global model configuration, executor tool configuration

**Runtime-context component**:
The named language-model component that generates and compacts Observations for tool-capable execution modes. It may use a separately configured OpenAI-compatible provider and Structured-output mode, while ReAct and the Plan-step Executor retain ownership of their tool loops.
_Avoid_: tool-loop model, observation provider

**Text stream**:
An asynchronous sequence containing only successive generated text deltas; it ends after the model response completes.
_Avoid_: response stream, event stream

**Event stream**:
An asynchronous sequence of OpenAI SDK `ResponseStreamEvent` values, including text, tool-call, reasoning, and completion usage information.
_Avoid_: raw stream, normalized event stream

**Model request**:
The bounded request submitted to an OpenAI-compatible provider: input plus optional instructions and tool definitions.
_Avoid_: provider request, completion request

**Tool execution**:
The upper-layer responsibility for acting on a model-issued function call and deciding whether to make a follow-up model request.
_Avoid_: automatic tool loop, LLM tool execution

**Execution mode**:
A whole-task answer strategy selected explicitly through a common factory, and eventually by a deterministic router. The modes are `direct`, `tool_agent`, `react`, and `plan_execute`; an Execution mode is not an Executor.
_Avoid_: executor, agent architecture

**Execution answer**:
The common whole-task result returned by an Execution mode: an answer plus its `completed` or `failed` terminal status and, on failure, an error. It is distinct from an Agent result and an Execution result.
_Avoid_: Agent result, step result

**Ephemeral execution**:
A whole-task Execution mode invocation that does not create Checkpoints and cannot be resumed. `direct`, `tool_agent`, and `react` are Ephemeral executions.
_Avoid_: durable run, resumable execution

**Direct mode**:
An Ephemeral Execution mode that obtains an answer from one language-model call without exposing tools. A nonempty response is a completed Execution answer.
_Avoid_: tool-free executor

**Tool-agent mode**:
An Execution mode limited to one language-model request and at most one Tool execution. When a model requests more than one tool, only its first request is executed. A tool's host-rendered result is the final answer rather than a second model-generated summary; when no tool is selected, the model's text is the final answer. A tool or rendering failure is a failed Execution answer.
_Avoid_: single-round ReAct, executor

**ReAct mode**:
An Ephemeral Execution mode that iterates language-model requests and Tool executions toward one answer, without a Planner or Plan. It ends only when its final structured response explicitly declares the goal satisfied.
_Avoid_: Executor, plan execution

**Plan–execute mode**:
An Execution mode in which an Agent obtains a Plan from a Planner and completes its Plan steps through the Plan-step Executor. Its Execution answer is the last completed Plan step's handoff.
_Avoid_: ReAct mode, executor

**Plan**:
A versioned, machine-validatable execution contract produced by a Planner for the whole goal, consisting of ordered Plan steps plus their completion criteria. Replanning appends an immutable revision rather than overwriting prior Plans.
_Avoid_: prompt, free-form plan

**Plan revision**:
One immutable version of the Plan for a goal. A goal permits at most three revisions, including the initial Plan.
_Avoid_: overwrite, mutable plan

**Plan step**:
One sequentially executed unit of a Plan, with its intended outcome and completion criterion; it does not prescribe tool calls.
_Avoid_: task, instruction, tool invocation

**Step execution**:
The latest attempt to complete one Plan step in a Plan revision, recorded in Agent state by revision and step ID without changing the Plan. Its status is pending, running, completed, failed, skipped, or interrupted; an interrupted execution may be restarted as a fresh attempt that reconciles the intended outcome without treating unconfirmed prior tool output as fact.
_Avoid_: plan mutation, replanning, continued transcript

**Execution attempt**:
One immutable, numbered effort to complete a specific Plan step. A new attempt begins when an interrupted Step execution is recovered; completed Plan steps do not receive another attempt.
_Avoid_: transcript continuation, overwritten execution

**Agent**:
The top-level coordinator that owns the Plan–Execute loop, records execution results in state, and decides whether to invoke the Planner again.
_Avoid_: executor, planner, worker

**Agent state**:
The durable coordination record supplied to the Planner: the goal, current Plan revision and position, execution results, and optional memory summary. It is captured as Checkpoints for an Agent run.
_Avoid_: session, context window

**Agent run**:
One identified invocation of the top-level Agent, addressed through the CLI by an application-generated UUIDv4 run ID and resumable from its stored Checkpoints. Only one active execution may own an Agent run under a renewable 60-second lease.
_Avoid_: session, thread

**Checkpoint**:
One durable snapshot of a LangGraph execution within an Agent run, stored in MySQL for either the top-level Agent graph or a Step-execution graph. It contains unredacted state and is resumed only from its latest version.
_Avoid_: savepoint, snapshot

**Persistence failure**:
The condition in which MySQL cannot record the next Checkpoint; it ends the Agent run before any further Tool execution.
_Avoid_: degraded mode, best-effort persistence

**Run lease**:
A renewable 60-second MySQL claim granting one process exclusive authority to execute or resume an Agent run. It is renewed every 15 seconds and evaluated by MySQL server time.
_Avoid_: lock, active session

**Recovery decision**:
The caller's disposition of an interrupted Step execution: start a fresh recovery attempt, record it as failed for replanning, or abort the Agent run as blocked.
_Avoid_: transcript continuation, crash recovery

**Idempotent tool**:
An MCP tool whose host registration explicitly declares that repeating the same invocation does not create an additional externally observable effect. Idempotency may inform an Executor's reconciliation strategy, but does not determine whether an interrupted Step execution can start a fresh recovery attempt.
_Avoid_: recovery gate, retryable step

**Blocked result**:
The terminal Agent result emitted when its Plan-revision budget is exhausted without completing the goal, or when the caller aborts an interrupted Agent run; it retains the Plan and execution history.
_Avoid_: retry, silent failure

**Run registry**:
The project-owned MySQL record of an Agent run's identity, terminal status, configuration fingerprint, and Run lease. It does not duplicate Checkpoint state.
_Avoid_: checkpoint table, session registry

**Planner**:
The component that derives a complete Plan from the goal and Agent state without directly accessing tools or the environment.
_Avoid_: executor, tool caller

**Executor**:
The component that makes one bounded Tool execution attempt for a Plan step, selecting from tools exposed by its host without changing the Plan or requesting a new Plan.
_Avoid_: agent, planner, automatic tool loop

**MCP tool set**:
The named, host-configured collection of Model Context Protocol tools made available to an Executor for a Step execution.
_Avoid_: tool permissions, tool server

**Tool runtime**:
The shared host-configured MCP connection, tool set, allowlist, and forced working-directory boundary used by tool-enabled Execution modes. It owns one MCP session for a mode invocation.
_Avoid_: tool client, per-mode tool configuration

**Tool-result renderer**:
The deterministic conversion of a Tool runtime result into an Execution answer without another language-model request. It preserves text, stably serializes structured or multi-content values, and rejects values that cannot be presented.
_Avoid_: tool-result summary, model synthesis

**Execution result**:
The concise final explanation returned by an Executor after a completed Step execution, specifically stating how the Plan step's completion criterion was met.
_Avoid_: transcript, tool trace

**Step context**:
The durable, cumulative handoff from successful Step executions to the next Executor invocation: the files read, files modified, and observations gathered during the Agent run. Its file paths are normalized, relative POSIX paths within the Agent working directory.
_Avoid_: message transcript, tool trace

**Context update**:
The strict structured portion of a successful Step execution's completion receipt that extends Step context with files read, files modified, and observations attributed to that Plan step.
_Avoid_: inferred tool trace, raw tool output

**Step recovery record**:
The durable recovery-facing record for one Execution attempt, pairing its Step execution with the Context update produced on successful completion. Its immutable history is used to reconstruct trustworthy Step context when an interrupted run is resumed.
_Avoid_: checkpoint internals, message transcript, tool trace

**Tool-calling model**:
A LangChain chat model configured for an OpenAI-compatible provider that can request tools from an MCP tool set during Step execution.
_Avoid_: text stream, planner model

**Tool-call batch**:
All function calls emitted in one response by a Tool-calling model. An Executor starts every call in the batch concurrently, waits until every call has settled, and returns their results in request order before continuing the Step execution.
_Avoid_: parallel Plan execution, tool-call sequence

**Tool round**:
One tool-capable Model request together with the resulting Tool-call batch, if any. A final no-tool completion request is not a Tool round.
_Avoid_: individual tool call, model turn

**Raw tool result**:
The complete request parameters and ordered results, including errors, for one Tool round's Tool-call batch. Raw tool results are transient Model context, never Step context or Durable State; complete diagnostic tracing may record them outside that state.
_Avoid_: durable tool trace, observation

**Observation**:
A schema-validated, concise historical record generated after one Tool round. It records the round number plus tool-call-ID-attributed confirmed facts, reported errors, and model inferences; generation or validation failure fails the active execution.
_Avoid_: raw tool output, durable fact

**Runtime context window**:
The context supplied to every Model request within a tool loop: the three most recent Tool rounds' Raw tool results, compressed Observations for earlier Tool rounds, and the applicable Durable State. It has an explicit token budget, defaulting to 128,000 tokens and configurable per component; if compaction cannot fit the required Durable State and Raw tool results, execution fails.
_Avoid_: Durable State, message transcript
