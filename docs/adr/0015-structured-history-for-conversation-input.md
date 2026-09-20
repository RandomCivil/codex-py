# Structured history for Conversation input

Every Conversation turn supplies the selected Execution mode with the complete preceding Conversation history as stable structured JSON and supplies the current user input separately. The history contains the public turn fields—sequence, user input, selected mode, run ID, status, and answer or error—but excludes checkpoints, Plans, step state, and raw tool output.

## Considered Options

- Serialize history as an unstructured transcript or concatenate it into the current goal — rejected because it loses field boundaries and confuses historical context with the new request.
- Expose internal Agent and tool state in history — rejected because unconfirmed or implementation-specific state must remain behind the existing recovery and diagnostic boundaries.
