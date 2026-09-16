# Codex Python

This context coordinates language-model interactions for the Codex Python agent.

## Language

**OpenAI-compatible provider**:
A service exposing the OpenAI Responses API through a caller-supplied base URL, API key, and model name.
_Avoid_: OpenAI provider, model provider

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
The Executor's single attempt to complete one Plan step in a Plan revision, recorded in Agent state by revision and step ID without changing the Plan. Its status is pending, running, completed, failed, or skipped; a failed step is returned to the Planner for revision.
_Avoid_: plan mutation, replanning, autonomous recovery

**Agent**:
The top-level coordinator that owns the Plan–Execute loop, records execution results in state, and decides whether to invoke the Planner again.
_Avoid_: executor, planner, worker

**Agent state**:
The durable coordination record supplied to the Planner: the goal, current Plan revision and position, execution results, and optional memory summary.
_Avoid_: session, context window

**Blocked result**:
The terminal Agent result emitted when its Plan-revision budget is exhausted without completing the goal; it retains the Plan and execution history.
_Avoid_: retry, silent failure

**Planner**:
The component that derives a complete Plan from the goal and Agent state without directly accessing tools or the environment.
_Avoid_: executor, tool caller

**Executor**:
The component that makes one bounded Tool execution attempt for a Plan step, selecting from tools exposed by its host without changing the Plan or requesting a new Plan.
_Avoid_: agent, planner, automatic tool loop

**MCP tool set**:
The named, host-configured collection of Model Context Protocol tools made available to an Executor for a Step execution.
_Avoid_: tool permissions, tool server

**Execution result**:
The concise final explanation returned by an Executor after a completed Step execution, specifically stating how the Plan step's completion criterion was met.
_Avoid_: transcript, tool trace

**Tool-calling model**:
A LangChain chat model configured for an OpenAI-compatible provider that can request tools from an MCP tool set during Step execution.
_Avoid_: text stream, planner model
