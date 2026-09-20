# Persistent Conversations as Agent-run aggregates

Multi-turn interaction is modeled as a MySQL-persisted Conversation addressed by `conv_id`, with each strictly ordered Conversation turn creating and retaining a distinct Agent run. A Conversation is deliberately separate from an Agent run because Agent runs have single-goal state, terminality, recovery, and checkpoint semantics that cannot safely be reused for a later user turn.

## Considered Options

- Reuse one Agent run for the entire conversation — rejected because `AgentState` and immutable Plan history bind a run to one goal, and terminal runs cannot accept a new goal.
- Keep conversation state only in process memory — rejected because it would lose the durable/recoverable lifecycle already expected of Agent runs.
