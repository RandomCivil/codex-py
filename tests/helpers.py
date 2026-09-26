"""Small controlled doubles and protocol builders shared by execution-mode tests."""

import json


def react_decision(status, field=None, value=None):
    """Build a valid ReAct decision response for a controlled model double."""
    lines = ["BEGIN REACT_DECISION", f"STATUS={json.dumps(status)}"]
    if field is not None:
        lines.append(f"{field}={json.dumps(value)}")
    return "\n".join([*lines, "END REACT_DECISION"])


class ToolModel:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []
        self.tools = None
        self.request_tools = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    async def ainvoke(self, messages):
        self.requests.append(list(messages))
        self.request_tools.append(self.tools)
        return next(self.responses)


class Runtime:
    def __init__(self, result=None):
        self.tools = []
        self.calls = []
        self.result = {"value": "done"} if result is None else result

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def invoke(self, call):
        self.calls.append(call)
        return self.result
