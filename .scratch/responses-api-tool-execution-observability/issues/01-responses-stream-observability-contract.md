# 01: Complete the Responses stream observability contract

**What to build:** Every streamed Responses API request produces safe, independently attributable LLM timing and provider-backed usage logs. The logs distinguish successful and failed requests, retain machine-readable nulls when a metric cannot be known, and stay out of the CLI's final stdout result.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] A completed streamed response reports first-text latency, completion duration, and provider usage under its existing component attribution.
- [ ] Generation-speed metrics use the approved first-to-last text-delta formulas and are null for no-text or zero/one-token output.
- [ ] A failed or interrupted response reports `request_status=failed` timing without inventing usage, and trace output never changes the final CLI stdout result.
