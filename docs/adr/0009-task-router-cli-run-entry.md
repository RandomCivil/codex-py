# Task Router at the CLI run entry point

**Status: accepted.** A new CLI `run` first creates a Task Analyzer from the
explicit Task Analyzer component provider configuration, then invokes the Task
Router exactly once before any Execution mode begins. The Task Analyzer
configuration inherits the effective Planner configuration unless YAML supplies
a `task_analyzer` override. The router uses the existing Execution-mode factory:
direct, tool-agent, and ReAct receive their existing configuration and Tool
runtime; Plan–execute receives the existing durable Agent.

The CLI returns the selected Execution mode, the unchanged Execution answer,
the optional Task Analysis, and an optional safe analysis error. The existing
run ID and registry lifecycle remain in place so terminal router outcomes can
be reported consistently. `resume` is deliberately not routed: it restores an
existing Plan–execute Agent run and never re-analyzes a goal or changes its
selected Execution mode.

This extends ADR-0008's previously programmatic-only composition point. It
preserves ADR-0007's rule that the Plan-step Executor is exclusive to
Plan–execute and that selected-mode failures never automatically escalate.
