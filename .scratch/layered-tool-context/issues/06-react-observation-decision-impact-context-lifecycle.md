# 06: ReAct Observation decision-impact context lifecycle

**What to build:** A ReAct tool loop can decide, for every successful observation-class Tool call, whether its resulting Observation changes the active decision, and then retain the most useful form of evidence without blocking the loop. A positive decision-impact Observation replaces its Raw fallback; a negative one remains Raw only through its normal corrective window before leaving Runtime context, while complete provenance remains diagnostically available.

**Blocked by:** 01: Shared Runtime context policy; 02: ReAct layered Tool context.

**Status:** ready-for-agent

- [ ] A ReAct observation request receives its deterministic decision summary and returns schema-validated decision impact plus affected targets for every existing successful observation-class call, including observation-class `exec`.
- [ ] A positive impact replaces Raw evidence; a negative impact retains Raw evidence until its existing window expires; a pending, failed, or invalid impact judgment preserves Raw fallback without delaying the ReAct loop.
- [ ] ReAct-facing tests cover every impact outcome, target invariants, target-union behavior during Observation merging, native-read exclusions, and unchanged diagnostic provenance.
