# Conversation-owned recovery

`conversation resume` recovers the Active Conversation turn through the existing Agent-run recovery path and then records the reconciled result in its Conversation. Standalone `agent resume` remains supported; a later Conversation operation reconciles its linked turn before allowing an append, so the legacy command does not need to own conversation state.

## Considered Options

- Remove or alter standalone `agent run` and `agent resume` — rejected because they remain the compatible interface for one-off Agent runs.
- Require all associated runs to be resumed only through Conversation commands — rejected because existing operational recovery remains valid and must not be broken.
