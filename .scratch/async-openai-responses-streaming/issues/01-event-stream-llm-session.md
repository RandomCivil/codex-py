# 01 — Event Stream LLM Session

**What to build:** An agent developer can configure an OpenAI-compatible provider and use a reusable asynchronous LLM session to submit a Model request and observe every SDK Event stream value, including reasoning, tool-call, completion, and usage information. The upper layer retains all Tool execution and follow-up request decisions.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] A configured asynchronous LLM session submits string and input-item-sequence Model requests with optional instructions and tool definitions, using the selected model and provider configuration.
- [ ] Consumers receive SDK `ResponseStreamEvent` values in provider order without event normalization or dropped tool-call/reasoning/completion data.
- [ ] The session supports asynchronous context management and explicit close, correctly releases stream resources, propagates SDK errors unchanged, and makes no automatic retry.
