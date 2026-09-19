# 02: Separate Runtime-context contracts from dynamic evidence

**What to build:** The Runtime-context component sends a stable, Structured-output-mode-appropriate Observation or Observation-merge contract ahead of the changing source evidence, so repeated structured Model requests retain a clear reusable prefix while Observations remain schema-validated and attributable.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] Observation generation and Observation merging use distinct fixed contracts followed by only their applicable dynamic Raw tool results or Observations; strict-schema contracts omit only schema-enforced repetition while JSON-object contracts retain the textual shape guarantees needed for compatibility.
- [ ] Existing Structured-output mode selection, expected-round validation, source Tool-call-ID preservation, Raw tool-result transience, and failure behavior remain unchanged.
