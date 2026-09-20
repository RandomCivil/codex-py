# Multi-turn Conversations over existing Execution modes

Status: ready-for-agent

## Problem Statement

The application can execute one goal through an existing Execution mode and, for Plan–execute, recover one Agent run. It cannot preserve a user's earlier inputs and answers when they continue an interaction in a later invocation. Callers need a backend-only, durable Conversation addressed by `conv_id`, where each new turn can use all earlier turns as context while retaining the existing Agent-run, recovery, and Execution-mode boundaries.

## Solution

Add a MySQL-backed Conversation application service and CLI commands for creating or continuing a Conversation, recovering its Active Conversation turn, and reading its history. A Conversation is a durable ordered aggregate identified by `conv_id`; every Conversation turn owns a distinct `run_id` and explicitly invokes one existing Execution mode. The complete structured Conversation history is authoritative and becomes context for the next turn, while its current user input remains distinct. Existing `agent run` and `agent resume` retain their single-run behavior.

## User Stories

1. As a CLI caller, I want to start a Conversation without supplying `conv_id`, so that the application creates one for my first turn.
2. As a CLI caller, I want the newly created `conv_id` returned in stable JSON, so that I can continue the same Conversation later.
3. As a CLI caller, I want to supply an existing `conv_id` with a new input, so that the application appends a Conversation turn rather than treating it as a separate interaction.
4. As a CLI caller, I want a nonexistent explicit `conv_id` rejected, so that a typographical error cannot silently create a forked Conversation.
5. As a CLI caller, I want every Conversation turn assigned its own Agent-run ID, so that I can inspect and recover its underlying work through the established run mechanisms.
6. As a CLI caller, I want to choose `direct`, `tool_agent`, `react`, or `plan_execute` for each turn, so that I can use an existing Execution mode appropriate to that turn.
7. As a CLI caller, I want the selected Execution mode retained with the turn, so that the Conversation history explains how each answer was produced.
8. As a user continuing a Conversation, I want all preceding turns supplied as structured context, so that a later Agent invocation can account for prior questions, answers, failures, and blocked work.
9. As a user, I want my current input kept separate from preceding history, so that the Agent can distinguish my new request from historical context.
10. As a user, I want a completed turn's final answer retained, so that it is visible to me and usable by later turns.
11. As a user, I want a failed turn's safe error retained, so that I can correct the request in a subsequent turn.
12. As a user, I want a blocked turn distinguished from a failed turn, so that I can understand that the Agent reached a normal blocked outcome rather than a technical failure.
13. As a user, I want to continue a Conversation after a terminal failed or blocked turn, so that one unsuccessful request does not discard the interaction.
14. As a caller, I want a Conversation to accept only one Active Conversation turn at a time, so that every turn receives one unambiguous, complete preceding history.
15. As a caller that submits concurrently, I want a stable `conversation busy` result that identifies the active work, so that I can wait or retry without creating an ordering race.
16. As an operator, I want a turn durably recorded before its Agent run begins, so that a crash cannot leave executed work unlinked from its Conversation.
17. As an operator, I want a crash-interrupted durable Plan–execute turn to remain recoverable, so that existing checkpoint and Agent-run recovery continue to work.
18. As an operator, I want `conversation resume` to recover the Active Conversation turn and record its reconciled result, so that recovery restores both the Agent run and Conversation view.
19. As an operator, I want a crash-interrupted ephemeral turn marked terminally failed before another turn is accepted, so that an unrecoverable in-process execution does not permanently block a Conversation.
20. As an operator, I want a later Conversation operation to reconcile a turn whose associated run was independently resumed through `agent resume`, so that the legacy recovery command does not leave stale conversation state.
21. As a CLI caller, I want `conversation show --conv-id` to return complete ordered structured history, so that I can retrieve a Conversation without submitting a new turn.
22. As a CLI caller, I want `conversation run` and `conversation resume` to return the current turn result rather than the entire growing history, so that normal responses remain bounded by the current operation.
23. As a maintainer, I want Conversation history to exclude Checkpoints, Plans, Step executions, and raw tool results, so that only public cross-turn context crosses the established recovery and diagnostic boundaries.
24. As a maintainer, I want every turn's effective non-secret configuration preserved through its existing Agent run, so that different turns may use different allowed configurations without storing credentials in Conversation data.
25. As an existing CLI user, I want `agent run` and `agent resume` unchanged, so that adding Conversations does not alter one-off task workflows.

## Implementation Decisions

- Use the glossary terms Conversation, Conversation turn, Active Conversation turn, Conversation history, Conversation input, Conversation-turn result, and Conversation recovery. A Conversation is not an Agent run; an Agent run remains one invocation with its own goal, terminality, Checkpoints, and recovery behavior.
- Add a Conversation application service as the single highest-level feature seam. It owns Conversation creation, append serialization, history retrieval, turn lifecycle projection, reconciliation, and recovery composition. The CLI is a thin adapter over this service; existing Agent-run and Execution-mode seams remain downstream dependencies.
- Persist Conversations and turns in MySQL through project-owned, versioned, idempotent migrations. A Conversation has a UUIDv4 `conv_id`. A turn belongs to exactly one Conversation, has a strictly increasing sequence number, a distinct UUIDv4 `run_id`, a selected Execution mode, the current user input, a lifecycle status, and a terminal answer or safe error as applicable.
- `conversation run` accepts the normal explicit configuration and working-directory inputs, current user input, and an explicit `--execution-mode`. Omitting `--conv-id` atomically creates a Conversation and its first turn; providing it appends only to an existing Conversation. It returns stable JSON containing the `conv_id`, current turn identity, `run_id`, selected Execution mode, status, and final answer or error.
- `conversation show --conv-id` returns the complete ordered structured Conversation history. `conversation run` and `conversation resume` return only their current operation's turn result, not the complete history.
- A Conversation input consists of the entire preceding structured Conversation history and the current user input as separate values. The history fields are sequence, user input, selected Execution mode, run ID, status, and final answer or error. Render this stable structure into each selected Execution mode's model-facing request while retaining the current input separately; do not concatenate history into an Agent goal.
- Initially apply no context-budget, truncation, or compaction policy to Conversation history: every prior turn is persisted and supplied. If an OpenAI-compatible provider rejects an oversized request, the current turn records that failure. A future policy requires a separate decision.
- Reuse every existing Execution mode (`direct`, `tool_agent`, `react`, `plan_execute`) per turn. Each invocation receives a new run ID. Only Plan–execute retains its existing durable Checkpoint and recovery semantics; Direct, tool-agent, and ReAct remain Ephemeral executions.
- The turn lifecycle has nonterminal `pending` and `running` states and terminal `completed`, `failed`, and `blocked` results. The terminal status, answer, and error are projected from the selected mode/Agent-run outcome without converting a blocked outcome into failure.
- Serialize appends so that there is exactly one Active Conversation turn. An append collision returns `conversation busy` with the active turn/run identification; do not queue, silently retry, or execute turns in parallel. Persist the new turn before starting its Agent run.
- Before accepting a new append or performing recovery, reconcile the Active Conversation turn against its linked Agent run. An interrupted Plan–execute turn remains active until recovered through `conversation resume`, which delegates to established Agent-run recovery and then atomically records the turn result. An interrupted Ephemeral execution is terminally failed before a later append is permitted.
- Preserve standalone `agent run` and `agent resume` semantics. If standalone `agent resume` is used for a run linked to a Conversation, the Conversation service reconciles the resulting terminal run at the next Conversation operation; the legacy command is not required to manage Conversation records.
- Use a conversation-scoped concurrency mechanism or equivalent transactional conditional write so persistence establishes the one-Active-turn invariant across processes. Do not rely on the existing Agent-run lease alone, because it protects only a single run, not the aggregate append order.
- Retain effective provider and tool configuration through the existing per-run non-secret configuration snapshot/fingerprint. Conversation-level configuration is not pinned, and API keys must never be stored in Conversation data or output.
- Keep `agent run`, `agent resume`, and `agent migrate` compatible. Add separate `conversation run`, `conversation resume`, and `conversation show` commands; no UI, service authentication, conversation list, or search interface is introduced.
- Honor ADR-0011 through ADR-0015 in addition to the existing Agent, Execution-mode, persistence, recovery, and runtime-context ADRs.

## Testing Decisions

- Test at the Conversation application-service seam using controlled Execution-mode runners, a controlled Agent-run recovery dependency, and a controllable Conversation store. Good tests assert observable Conversation behavior and emitted result contracts; they do not assert SQL layout, prompt wording, private helper calls, graph topology, or Checkpoint internals.
- Verify creation without `conv_id` generates and returns a UUIDv4 Conversation identity, persists its first turn before runner invocation, and associates a distinct run ID.
- Verify continuation with an existing `conv_id` appends the next sequence exactly once and passes all prior public turn fields plus separate current input to the selected mode.
- Verify an explicit nonexistent or invalid Conversation ID returns the documented not-found/validation outcome without creating a Conversation.
- Verify every allowed explicit Execution mode is selected through the established factory seam, retains its mode on the turn, and receives an independent run ID. Verify that a turn never causes automatic mode routing, fallback, or escalation.
- Verify completed, failed, and blocked outcomes retain the correct status and answer/error; verify another turn can follow every terminal outcome and sees that outcome in history.
- Verify no raw tool result, Checkpoint, Plan, or Step-execution value becomes Conversation history or model-facing Conversation input.
- Verify that only one Active Conversation turn can exist under concurrent append attempts and that the losing operation returns `conversation busy` with safe active-work identity. Exercise the invariant at the persistence/service boundary rather than testing an implementation-specific lock.
- Verify crash/restart reconciliation behavior: durable active work blocks append until Conversation recovery; `conversation resume` delegates to normal Agent-run recovery and updates the turn; standalone-run recovery is reconciled before a later Conversation append; interrupted Ephemeral work becomes failed before append is allowed.
- Verify `conversation show` exposes complete ordered history and that run/resume responses expose the current turn, not the whole history.
- Add thin CLI contract tests for required arguments, optional `conv_id` creation/continuation behavior, stable JSON fields, validation/not-found/busy outcomes, and delegation to the Conversation service. Do not duplicate service behavior exhaustively at the CLI layer.
- Add MySQL integration tests, marked and conditionally enabled consistently with existing MySQL tests, for migrations and cross-process conditional append behavior. They must use the dedicated test URL and never fall back to the runtime database.
- Use existing CLI migration tests as prior art for argument parsing and stable JSON; existing task-routing and Execution-mode tests as prior art for controlled factory/runner seams; existing registry, durable, recovery, and MySQL integration tests as prior art for leases, terminal-result projection, migrations, and recovery safety.

## Out of Scope

- A web, desktop, or other graphical Conversation interface.
- Conversation listing, deletion, archival, renaming, search, cross-Conversation retrieval, retention policy, or export.
- Authentication, authorization, user ownership, tenancy, or treating a UUID as an access-control mechanism.
- Conversation-history token budgets, truncation, summarization, compaction, retrieval, or provider-specific context-limit mitigation.
- Any new model-routing policy, automatic Execution-mode selection, fallback, escalation, or a Conversation-specific Agent implementation.
- Altering the established meaning of Agent, Agent run, Agent state, Planner, Plan-step Executor, Checkpoint, Step recovery, or Execution mode.
- Making Direct, tool-agent, or ReAct checkpointed, resumable, or durable.
- Exposing raw tool results, internal Plans, Step executions, Checkpoints, or provider credentials through Conversation history.
- Changing Atom MCP, tool policies, OpenAI-compatible provider configuration rules, or existing standalone CLI contracts.

## Further Notes

The first release deliberately favors complete durable history and explicit caller-selected modes over a UI or automatic routing. Provider context overflow is a normal recorded turn failure until a separately designed history-budget policy exists. The durable Conversation aggregate and its append ordering are governed by ADR-0011 and ADR-0013; reuse of existing modes and structured input by ADR-0012 and ADR-0015; and recovery integration by ADR-0014. Existing ADRs on durable Agent runs, MySQL Checkpoints, configuration fingerprints, Execution modes, routing, and layered runtime context continue to apply without reinterpretation.
