# 03: Handle judge and Tool failures without false completion

**What to build:** The Plan-step Executor skips completion judgment and Observation requests for an entire batch if any Tool call fails. Failed batches remain correction context; malformed judgments cannot change completion state and provider failures become failed Step executions.

**Blocked by:** 01: Judge a Tool-backed Plan step.

**Status:** completed

- [x] Any failed call suppresses completion judgment and Observation requests for its entire batch; failed results remain correction messages for the Tool-calling model.
- [x] Duplicate, out-of-range, already-completed, evidence-free, malformed, and inconsistent judge outputs are rejected locally. One repair request carries the original context, rejected output, and validation error; a second invalid response leaves progress pending and continues within budget.
- [x] A judge provider failure or inability to fit required evidence in the Runtime-context budget fails the Step rather than silently truncating evidence or accepting completion.
- [x] Controlled public Executor behavior demonstrates mixed and all-failed batches, invalid-output repair, unchanged progress after failed repair, and failed outcomes that leave replanning to the Agent.

## Comments

- Implemented with TDD. Mixed/all-failed batches, local judgment validation and one repair, provider failure, and Runtime-context budget failure are covered. Full suite: 337 passed, 7 skipped.
