# Codex Python

This context coordinates language-model interactions for the Codex Python agent.

## Language

**Line Protocol**:
A strict UTF-8, line-oriented model-response representation in which a response is delimited by matching `BEGIN <TYPE>` and `END <TYPE>` lines. Scalar fields use `PATH=JSON_LITERAL`, dot-separated paths represent nested objects, repeated paths represent scalar arrays, and repeated nested blocks represent object arrays; components use it where they require a structured response contract and validate it locally before conversion to a component's domain contract.
_Avoid_: JSON mode, JSON Schema, JSON object

**No-tool response**:
A legacy empty `NO_TOOL` Line Protocol block that formerly indicated a tool-capable model declined native tool calls. ReAct and the Plan-step Executor now treat a no-tool response's content as their terminal result.
_Avoid_: required no-tool marker, separate completion request

**OpenAI-compatible provider**:
A service exposing the OpenAI Responses API through a caller-supplied base URL, API key, and model name.
_Avoid_: OpenAI provider, model provider

**Component provider configuration**:
The effective OpenAI-compatible provider credentials, endpoint, and model name used by a named language-model component. Components may inherit shared values, then apply component-specific overrides.
_Avoid_: global model configuration, executor tool configuration

**Runtime-context component**:
The named language-model component that generates and compacts Observations for tool-capable execution modes. It may use a separately configured OpenAI-compatible provider, while ReAct and the Plan-step Executor retain ownership of their tool loops.
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
An Ephemeral Execution mode that iterates language-model requests and Tool executions toward one answer, without a Planner or Plan. It ends when a model response has no native tool calls; that response's content is the final answer.
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

**Conversation**:
The durable, user-facing sequence of Conversation turns for one continuing interaction, identified by a caller-supplied `conv_id`. A Conversation aggregates Agent runs but is not an Agent run.
_Avoid_: session, run, thread

**Interactive Conversation**:
A terminal-mediated Conversation in which one completed turn's result is presented before the next user input is accepted; it either creates a new Conversation or reconnects to one by `conv_id`.
_Avoid_: interactive session, chat session

**Conversation turn**:
One strictly ordered user input and its resulting invocation of a selected existing Agent implementation within a Conversation. Every Conversation turn has its own Agent run.
_Avoid_: Agent run, message, request

**Active Conversation turn**:
The sole nonterminal Conversation turn allowed in a Conversation. It prevents another turn from being appended until it becomes terminal or is explicitly recovered according to its selected Execution mode.
_Avoid_: queued message, parallel turn

**Conversation recovery**:
The operation that resumes the Agent run of an Active Conversation turn and records its reconciled terminal result in the Conversation. It is distinct from standalone Agent-run recovery.
_Avoid_: retrying a message, transcript continuation

**Conversation history**:
The complete structured, ordered record of a Conversation's turns, including their user inputs and results. It is the authoritative cross-turn record and is initially supplied without a context-budget limit.
_Avoid_: Agent state, checkpoint, transcript window

**Conversation-turn result**:
The terminal outcome retained for a Conversation turn: `completed`, `failed`, or `blocked`, with its final answer or error. `pending` and `running` are nonterminal turn states only.
_Avoid_: execution trace, checkpoint state

**Conversation input**:
The current user input paired with its Conversation history, supplied to the selected Execution mode for one Conversation turn. It preserves the user's input as distinct from prior turns and from an Agent run's goal.
_Avoid_: concatenated goal, Agent state

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
The component that makes one bounded Tool execution attempt for a Plan step, selecting from tools exposed by its host without changing the Plan or requesting a new Plan. When the model stops requesting tools, its nonempty content is the Plan-step handoff.
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
The complete request parameters and result, including an error where present, for one Tool call. Raw tool results are transient Model context, never Step context or Durable State; complete diagnostic tracing may record them outside that state.
_Avoid_: durable tool trace, observation

**File-grouped native-read rendering**:
The Runtime-context presentation of permanent-raw native `grep` and `read_file` evidence, grouped by resolved file path across Tool rounds with each result kept in round and request order. A call whose path cannot be resolved, or which reports an error, is retained in the `(unfiled)` group with its native request and complete result.
_Avoid_: raw-result compaction, observation

**Observation**:
A schema-validated, concise historical record generated asynchronously for one successful observation-class Tool call. It records the call's Tool round, tool-call ID, evidence, and decision impact; an impact-positive Observation replaces its Raw tool result, while an impact-negative call retains Raw evidence through its existing window and then leaves Runtime context. A failed Tool call never starts an Observation request and remains Raw evidence.
_Avoid_: raw tool output, durable fact

**Observation decision impact**:
The Observation-owned, explainable assessment of whether one successful observation-class Tool call changes the active tool-loop decision. It contains `affects_current_decision` and its affected targets: confirmed facts, summary, and/or Durable State.
_Avoid_: command intent, implicit context refresh

**Observation outcome**:
A transient diagnostic record for one asynchronous Observation attempt. It identifies the Tool call and its Tool round, whether the attempt is pending, succeeded, failed at the provider, was invalid, or was cancelled, and may record the error; it is not Model evidence, Step context, or Durable State.
_Avoid_: Observation, durable task record

**Tool-call evidence policy**:
The per-call rule that selects raw evidence, an Observation, or no retained evidence for a Runtime context window. `list_dir` and `glob` retain raw evidence only for the newest Tool round; `grep` and `read_file` retain raw evidence for the whole invocation; successful observation-class calls use asynchronous decision-impact Observations with Raw tool-result fallback, while failed calls remain Raw and never invoke the Runtime-context component.
_Avoid_: round-level context policy, universal observation policy

**Exec command class**:
The conservative static classification of an `exec` command under the Tool-call evidence policy. Only whitelisted, pure top-level read commands map to a read class; known writing commands and every compound, dynamically expanded, unknown, or unclassifiable command map to the write class.
_Avoid_: command intent, heuristic safety classification

**Runtime context window**:
The context supplied to every Model request within a tool loop: retained evidence selected independently for every Tool call by the Tool-call evidence policy, plus applicable Durable State. Its rendered Goal is a distinct final section, while the durable goal remains part of Agent state. It has an explicit token budget, defaulting to 128,000 tokens and configurable per component; if compaction cannot fit Durable State and required Raw tool results, execution fails.
_Avoid_: Durable State, message transcript
