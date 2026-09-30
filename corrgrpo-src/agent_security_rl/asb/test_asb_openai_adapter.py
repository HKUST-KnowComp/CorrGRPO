#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from aios.llm_core.llm_classes.gpt_llm import GPTLLM


class FakeLogger:
    def log(self, *_args, **_kwargs):
        pass


class FakeCompletions:
    def __init__(self, message):
        self.message = message
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=self.message)]
        )


class FakeAgentProcess:
    def __init__(self, query):
        self.query = query
        self.agent_name = "adapter-test"
        self.response = None

    def set_status(self, _status):
        pass

    def set_start_time(self, _value):
        pass

    def set_end_time(self, _value):
        pass

    def set_response(self, response):
        self.response = response


def make_llm(message):
    llm = object.__new__(GPTLLM)
    llm.model_name = "local-test"
    llm.max_new_tokens = 128
    llm.logger = FakeLogger()
    completions = FakeCompletions(message)
    llm.model = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    return llm, completions


def test_workflow_json_cleanup():
    message = SimpleNamespace(
        content=(
            "Here is the plan:\n```json\n"
            '[{"message":"first","tool_use":["safe_tool"]},'
            '{"message":"finish","tool_use":[]}]\n```'
        ),
        reasoning_content=None,
        tool_calls=None,
    )
    llm, _ = make_llm(message)
    query = SimpleNamespace(
        messages=[{"role": "user", "content": "plan"}],
        tools=None,
        message_return_type="json",
    )
    process = FakeAgentProcess(query)
    llm.process(process)
    assert process.response.response_message.startswith('[{"message"')


def test_text_tool_fallback():
    message = SimpleNamespace(
        content='[{"message":"use it","tool_use":["safe_tool"]}]',
        reasoning_content=None,
        tool_calls=None,
    )
    llm, completions = make_llm(message)
    query = SimpleNamespace(
        messages=[{"role": "user", "content": "execute"}],
        tools=[{
            "type": "function",
            "function": {"name": "safe_tool", "description": "test"},
        }],
        message_return_type="text",
    )
    process = FakeAgentProcess(query)
    old_mode = os.environ.get("ASB_OPENAI_TOOL_MODE")
    os.environ["ASB_OPENAI_TOOL_MODE"] = "text"
    try:
        llm.process(process)
    finally:
        if old_mode is None:
            os.environ.pop("ASB_OPENAI_TOOL_MODE", None)
        else:
            os.environ["ASB_OPENAI_TOOL_MODE"] = old_mode
    assert process.response.tool_calls == [
        {"message": "use it", "tool_use": ["safe_tool"]}
    ]
    assert "tools" not in completions.request
    assert "Available tools" in completions.request["messages"][-1]["content"]
    assert query.messages[-1]["content"] == "execute"


def test_ellipsis_tool_arguments():
    message = SimpleNamespace(
        content="[{'name': 'safe_tool', 'parameters': {...}}]",
        reasoning_content=None,
        tool_calls=None,
    )
    llm, _ = make_llm(message)
    query = SimpleNamespace(
        messages=[{"role": "user", "content": "execute"}],
        tools=[{
            "type": "function",
            "function": {"name": "safe_tool", "description": "test"},
        }],
        message_return_type="text",
    )
    process = FakeAgentProcess(query)
    llm.process(process)
    assert process.response.tool_calls == [
        {"name": "safe_tool", "parameters": {}}
    ]


def test_agentdojo_inference_parameters():
    message = SimpleNamespace(
        content="done",
        reasoning_content=None,
        tool_calls=None,
    )
    llm, completions = make_llm(message)
    query = SimpleNamespace(
        messages=[{"role": "user", "content": "answer"}],
        tools=None,
        message_return_type="text",
    )
    process = FakeAgentProcess(query)
    settings = {
        "ASB_TEMPERATURE": "0.0",
        "ASB_TOP_P": "0.9",
        "ASB_TOP_K": "20",
        "ASB_REPETITION_PENALTY": "1.05",
        "ASB_SEED_MODE": "random",
        "ASB_ENABLE_THINKING": "0",
    }
    old_values = {key: os.environ.get(key) for key in settings}
    os.environ.update(settings)
    try:
        llm.process(process)
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    request = completions.request
    assert request["temperature"] == 0.0
    assert request["top_p"] == 0.9
    assert 0 <= request["seed"] <= 1000000
    assert request["extra_body"] == {
        "top_k": 20,
        "repetition_penalty": 1.05,
        "chat_template_kwargs": {"enable_thinking": False},
    }


if __name__ == "__main__":
    test_workflow_json_cleanup()
    test_text_tool_fallback()
    test_ellipsis_tool_arguments()
    test_agentdojo_inference_parameters()
    print("ASB OpenAI adapter tests passed")
