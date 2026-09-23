"""Small controlled doubles shared by execution-mode tests."""


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
