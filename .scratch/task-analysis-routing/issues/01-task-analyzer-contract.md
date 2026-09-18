# 01: Task Analyzer contract

**What to build:** A caller can submit a nonempty goal to Task Analyzer and receive a strictly validated, descriptive task-characteristics result using the agreed prompt and Structured-output mode. Invalid model output is rejected clearly rather than being used to make an execution decision.

**Blocked by:** None (can start immediately).

**Status:** resolved

- [x] Task Analyzer returns exactly the agreed task-characteristics contract for valid structured model output in both supported Structured-output modes.
- [x] Malformed JSON, missing or extra fields, invalid categories, invalid score/count/boolean values, empty rationale, and model failures are rejected without producing an analysis.
- [x] Tests use controlled text-stream doubles and assert public request/result behavior without live provider dependencies.

## Answer

Implemented the Task Analyzer contract at the public `TaskAnalyzer.run(goal)` seam. Valid output is parsed into the exported `TaskAnalysis` value using either `json_schema` or `json_object`; invalid JSON, shape, categories, ranges, counts, booleans, rationale, empty goals, and provider failures are rejected without producing an analysis. Added controlled text-stream contract tests and exported `TaskAnalyzer`, `TaskAnalysis`, and `TaskAnalysisValidationError` from `agent`.

## Comments

- Verified with `poetry run pytest -q`: 162 passed, 5 skipped.
