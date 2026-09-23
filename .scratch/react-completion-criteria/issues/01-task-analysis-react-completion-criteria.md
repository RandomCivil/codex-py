# 01: Task Analysis supplies ReAct completion criteria

**What to build:** A routed ReAct execution receives an ordered collection of independently verifiable completion criteria from Task Analysis, while direct, tool-agent, and Plan–execute keep their existing contracts and deterministic routing remains code-owned.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] A valid ReAct Task Analysis contains ordered, nonempty, distinct completion criteria; non-ReAct analyses omit them, and local Line Protocol validation rejects invalid combinations and values.
- [ ] The Task Router passes the successful Task Analysis through its existing mode-factory seam without changing route selection or non-ReAct behavior.
- [ ] Controlled analyzer and router tests verify the ReAct routing boundary and criteria propagation without a live provider or Tool runtime.
