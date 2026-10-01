from io import StringIO
import json
import warnings

from openai._types import omit
from openai.lib.streaming.responses._responses import ResponseStreamState
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseOutputMessage,
)

from agent.trace import RunTrace


def test_trace_serializes_sdk_completed_stream_event_without_generic_warnings():
    response = Response.model_construct(id="resp_test", output=[])
    state = ResponseStreamState(input_tools=omit, text_format=omit)
    state.handle_event(
        ResponseCreatedEvent.model_construct(
            type="response.created", response=response, sequence_number=0
        )
    )
    message = ResponseOutputMessage.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": "BEGIN PLAN\nEND PLAN",
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        }
    )
    event = state.handle_event(
        ResponseCompletedEvent.model_construct(
            type="response.completed",
            response=response.model_copy(update={"output": [message]}),
            sequence_number=1,
        )
    )[0]
    output = StringIO()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Pydantic serializer warnings:")
        RunTrace(output).llm_event(event)

    payloads = [
        json.loads(line.split("data=", 1)[1])
        for line in output.getvalue().splitlines()
        if "data=" in line
    ]
    assert payloads[0]["type"] == "response.completed"
    assert payloads[0]["response"] == payloads[-1]
    content = payloads[0]["response"]["output"][0]["content"][0]
    assert content["text"] == "BEGIN PLAN\nEND PLAN"
    assert content["parsed"] is None
