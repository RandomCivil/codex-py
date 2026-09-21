import asyncio

from langchain_core.messages import AIMessage

from agent.execution import DirectMode, ExecutionAnswer, ReactMode, ToolAgentMode


class TextModel:
    def __init__(self, response): self.response = response; self.requests = []
    async def stream_text(self, input, **kwargs):
        self.requests.append((input, kwargs))
        yield self.response


class ToolModel:
    def __init__(self, *responses): self.responses = iter(responses); self.requests = []; self.tools = None
    def bind_tools(self, tools): self.tools = tools; return self
    async def ainvoke(self, messages):
        self.requests.append(messages)
        return next(self.responses)


class Runtime:
    def __init__(self): self.tools = []; self.calls = []
    async def __aenter__(self): return self
    async def __aexit__(self, *args): pass
    async def invoke(self, call): self.calls.append(call); return {"value": "done"}


def test_direct_mode_decodes_answer_and_prompts_for_line_protocol():
    model = TextModel('BEGIN ANSWER\nTEXT="Ready"\nEND ANSWER')
    answer = asyncio.run(DirectMode(model).run("Prepare"))
    assert answer == ExecutionAnswer("Ready", "completed")
    assert "BEGIN ANSWER" in model.requests[0][1]["instructions"]
    assert "text_format" not in model.requests[0][1]


def test_direct_mode_rejects_invalid_answer():
    assert asyncio.run(DirectMode(TextModel("Ready")).run("Prepare")).status == "failed"


def test_tool_agent_prompts_for_answer_and_decodes_no_tool_response():
    model = ToolModel(AIMessage(content='BEGIN ANSWER\nTEXT="No tool needed"\nEND ANSWER'))
    answer = asyncio.run(ToolAgentMode(model, Runtime()).run("Answer"))
    assert answer == ExecutionAnswer("No tool needed", "completed")
    assert "BEGIN ANSWER" in model.requests[0][0].content


def test_tool_agent_executes_a_tool_call_with_accompanying_text():
    model = ToolModel(AIMessage(content='BEGIN ANSWER\nTEXT="bad"\nEND ANSWER', tool_calls=[{"name": "x", "args": {}, "id": "1"}]))
    runtime = Runtime()
    answer = asyncio.run(ToolAgentMode(model, runtime).run("Answer"))
    assert answer == ExecutionAnswer('{"value": "done"}', "completed")
    assert [{key: call[key] for key in ("name", "args", "id")} for call in runtime.calls] == [
        {"name": "x", "args": {}, "id": "1"}
    ]


def test_react_requires_no_tool_before_separate_goal_completion():
    model = ToolModel(
        AIMessage(content="BEGIN NO_TOOL\nEND NO_TOOL"),
        AIMessage(content='BEGIN GOAL_COMPLETION\nANSWER="Done"\nGOAL_SATISFIED=true\nEND GOAL_COMPLETION'),
    )
    answer = asyncio.run(ReactMode(model, Runtime()).run("Complete"))
    assert answer == ExecutionAnswer("Done", "completed")
    assert "NO_TOOL" in model.requests[0][0].content
    assert "Do not include the final answer" in model.requests[0][0].content
    assert "GOAL_COMPLETION" in model.requests[1][-1].content


def test_react_terminal_prompt_requires_a_json_encoded_answer_and_reports_protocol_errors():
    model = ToolModel(
        AIMessage(content="BEGIN NO_TOOL\nEND NO_TOOL"),
        AIMessage(
            content=(
                "BEGIN GOAL_COMPLETION\n"
                "ANSWER: A Markdown answer on raw lines\n\n"
                "- rather than a JSON string\n"
                "GOAL_SATISFIED=true\n"
                "END GOAL_COMPLETION"
            )
        ),
    )

    answer = asyncio.run(ReactMode(model, Runtime()).run("Complete"))

    assert answer == ExecutionAnswer(
        None,
        "failed",
        error=(
            "react model returned an invalid protocol response: "
            "invalid line or text outside a block"
        ),
    )
    terminal_instruction = model.requests[1][-1].content
    assert "ANSWER=<JSON string literal>" in terminal_instruction
    assert 'ANSWER="A complete answer"' in terminal_instruction


def test_react_rejects_prose_instead_of_no_tool():
    model = ToolModel(AIMessage(content="Done"))
    assert asyncio.run(ReactMode(model, Runtime()).run("Complete")).status == "failed"


def test_react_reports_the_protocol_failure_for_a_nonempty_no_tool_response():
    model = ToolModel(AIMessage(content="BEGIN NO_TOOL\n\nComponent summary"))

    answer = asyncio.run(ReactMode(model, Runtime()).run("Count components"))

    assert answer == ExecutionAnswer(
        None,
        "failed",
        error=(
            "react model returned an invalid protocol response: "
            "blank lines and comments are not permitted"
        ),
    )


def test_react_executes_tool_calls_with_accompanying_text():
    model = ToolModel(
        AIMessage(content="I'll inspect the source tree.", tool_calls=[{"name": "list_dir", "args": {}, "id": "1"}]),
        AIMessage(content="BEGIN NO_TOOL\nEND NO_TOOL"),
        AIMessage(content='BEGIN GOAL_COMPLETION\nANSWER="Done"\nGOAL_SATISFIED=true\nEND GOAL_COMPLETION'),
    )
    runtime = Runtime()
    answer = asyncio.run(ReactMode(model, runtime).run("Inspect"))
    assert answer == ExecutionAnswer("Done", "completed")
    assert [{key: call[key] for key in ("name", "args", "id")} for call in runtime.calls] == [
        {"name": "list_dir", "args": {}, "id": "1"}
    ]
