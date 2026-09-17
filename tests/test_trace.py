from io import StringIO
from types import SimpleNamespace

from agent.trace import RunTrace


def test_llm_event_prints_complete_event_payload():
    output = StringIO()
    trace = RunTrace(output)

    trace.llm_event(
        SimpleNamespace(
            type="response.completed",
            id="resp_123",
            status="completed",
            usage={"total_tokens": 7},
        )
    )

    assert output.getvalue().splitlines()[0] == (
        '[llm event] data={"id": "resp_123", "status": "completed", '
        '"type": "response.completed", "usage": {"total_tokens": 7}}'
    )
