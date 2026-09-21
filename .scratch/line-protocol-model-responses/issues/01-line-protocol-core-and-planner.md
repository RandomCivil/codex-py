# 01: Line Protocol core and Planner slice

**What to build:** A Planner can obtain one complete `PLAN` Line Protocol response, decode it through a strict shared codec, and produce the same validated Plan contract that downstream Plan–Execute behavior expects.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] A single shared Line Protocol implementation accepts the specified typed boundaries, JSON-literal scalars, nested object paths, scalar arrays, and object-array blocks while rejecting malformed or ambiguous documents.
- [ ] The codec registry explicitly maps `PLAN` and nested `STEP` fields without inferred field names or array plurality.
- [ ] Planner requests prompt for `PLAN`, omit provider-specific output formatting, and still reject a protocol response that cannot produce a valid Plan.
- [ ] Focused tests cover valid Plan decoding, relevant escaping/nesting/array cases, and invalid protocol or Plan-contract responses.
