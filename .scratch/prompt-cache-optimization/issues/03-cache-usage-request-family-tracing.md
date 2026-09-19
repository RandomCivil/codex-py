# 03: Attribute cache usage to Model request families

**What to build:** Operators can identify the language-model component and safe static request family associated with each input-token and cache-read usage record, allowing cache effectiveness to be compared without exposing Model input or secrets.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Run tracing retains existing input, output, total, cached, and reasoning token reporting while adding component attribution and a deterministic identifier derived only from cache-relevant static request shape.
- [ ] Both Responses-style cached-token metadata and LangChain/provider-style cache-read metadata remain supported, absent metrics remain non-fatal, and error-level tracing continues to expose token accounting without full request content.

