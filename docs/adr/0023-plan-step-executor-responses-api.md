# Plan-step Executor uses the Responses API

The Plan-step Executor sends operational, completion-judge, and judge-repair requests through the shared OpenAI Responses API adapter using non-streaming `responses.create` calls with reasoning effort set to `none`. This keeps all of the Executor's model requests on the project's Responses API path while preserving host-owned MCP execution through `ToolNode`; the adapter translates between Responses function calls and the Executor's internal message representation.
