# Conversation turns use existing Execution modes

Each Conversation turn explicitly selects one existing Execution mode—`direct`, `tool_agent`, `react`, or `plan_execute`—rather than adding a new conversation-specific Agent. The complete structured Conversation history and current user input are provided through a distinct Conversation-input contract, preserving an Agent run's single-goal state and making the selection auditable.

## Considered Options

- Restrict Conversation turns to Plan–execute — rejected because the existing modes are intentionally distinct whole-task strategies and callers must be able to select each per turn.
- Concatenate history into the Agent goal — rejected because it makes the user's current goal and prior context indistinguishable in plans and durable state.
