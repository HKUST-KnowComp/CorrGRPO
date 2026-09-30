import re
from .base_llm import BaseLLM
import copy
import os
import random
import time

# could be dynamically imported similar to other models
from openai import OpenAI

import openai

from pyopenagi.utils.chat_template import Response
import json

class GPTLLM(BaseLLM):

    def __init__(self, llm_name: str,
                 max_gpu_memory: dict = None,
                 eval_device: str = None,
                 max_new_tokens: int = 1024,
                 log_mode: str = "console"):
        super().__init__(llm_name,
                         max_gpu_memory,
                         eval_device,
                         max_new_tokens,
                         log_mode)

    def load_llm_and_tokenizer(self) -> None:
        client_kwargs = {
            "max_retries": int(os.environ.get("OPENAI_MAX_RETRIES", "8")),
            "timeout": float(os.environ.get("OPENAI_TIMEOUT", "180")),
        }
        if os.environ.get("OPENAI_API_KEY"):
            client_kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
        if os.environ.get("OPENAI_BASE_URL"):
            client_kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
        self.model = OpenAI(**client_kwargs)
        self.tokenizer = None

    def parse_native_tool_calls(self, tool_calls):
        if tool_calls:
            parsed_tool_calls = []
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                raw_args = tool_call.function.arguments or "{}"
                try:
                    function_args = json.loads(raw_args)
                except (TypeError, json.JSONDecodeError):
                    function_args = {}
                parsed_tool_calls.append(
                    {
                        "name": function_name,
                        "parameters": function_args
                    }
                )
            return parsed_tool_calls
        return None

    def process(self,
            agent_process,
            temperature=0.0
        ):
        """ wrapper around openai api """
        agent_process.set_status("executing")
        agent_process.set_start_time(time.time())
        messages = copy.deepcopy(agent_process.query.messages)
        tools = agent_process.query.tools
        message_return_type = agent_process.query.message_return_type
        tool_mode = os.environ.get("ASB_OPENAI_TOOL_MODE", "text").lower()
        if tool_mode not in {"text", "native", "hybrid"}:
            raise ValueError(
                "ASB_OPENAI_TOOL_MODE must be text, native, or hybrid"
            )
        request_tools = None
        if tools:
            if tool_mode in {"text", "hybrid"}:
                messages = self.tool_calling_input_format(messages, tools)
            if tool_mode in {"native", "hybrid"}:
                request_tools = tools
        # print(messages)
        self.logger.log(
            f"{agent_process.agent_name} is switched to executing.\n",
            level = "executing"
        )
        request_delay = float(os.environ.get("ASB_API_REQUEST_DELAY", "0"))
        if request_delay > 0:
            time.sleep(request_delay)
        try:
            effective_temperature = float(
                os.environ.get("ASB_TEMPERATURE", str(temperature))
            )
            seed_mode = os.environ.get("ASB_SEED_MODE", "fixed").lower()
            if seed_mode == "random":
                request_seed = random.randint(0, 1000000)
            elif seed_mode == "fixed":
                request_seed = int(os.environ.get("ASB_SEED", "0"))
            elif seed_mode == "omit":
                request_seed = None
            else:
                raise ValueError("ASB_SEED_MODE must be fixed, random, or omit")
            request = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": self.max_new_tokens,
                "temperature": effective_temperature,
            }
            if request_seed is not None:
                request["seed"] = request_seed
            if os.environ.get("ASB_TOP_P"):
                request["top_p"] = float(os.environ["ASB_TOP_P"])
            extra_body = {}
            if os.environ.get("ASB_TOP_K"):
                extra_body["top_k"] = int(os.environ["ASB_TOP_K"])
            if os.environ.get("ASB_REPETITION_PENALTY"):
                extra_body["repetition_penalty"] = float(
                    os.environ["ASB_REPETITION_PENALTY"]
                )
            if os.environ.get("ASB_ENABLE_THINKING"):
                extra_body["chat_template_kwargs"] = {
                    "enable_thinking": os.environ["ASB_ENABLE_THINKING"].lower()
                    in {"1", "true", "yes"}
                }
            if extra_body:
                request["extra_body"] = extra_body
            if request_tools is not None:
                request["tools"] = request_tools
            response = self.model.chat.completions.create(**request)
            message = response.choices[0].message
            response_message = (
                message.content
                or getattr(message, "reasoning_content", None)
            )
            tool_calls = self.parse_native_tool_calls(
                message.tool_calls
            )
            if tools and not tool_calls and response_message:
                tool_calls = super().parse_tool_calls(response_message)
            if not tools and message_return_type == "json":
                response_message = self.parse_json_format(response_message or "")
            # print(tool_calls)
            # print(response.choices[0].message)
            agent_process.set_response(
                Response(
                    response_message = response_message,
                    tool_calls = tool_calls
                )
            )
        except openai.APIConnectionError as e:
            agent_process.set_response(
                Response(
                    response_message = f"Server connection error: {e.__cause__}"
                )
            )
        except openai.RateLimitError as e:
            agent_process.set_response(
                Response(
                    response_message = f"OpenAI RATE LIMIT error {e.status_code}: (e.response)"
                )
            )
        except openai.APIStatusError as e:
            agent_process.set_response(
                Response(
                    response_message = f"OpenAI STATUS error {e.status_code}: (e.response)"
                )
            )
        except openai.BadRequestError as e:
            agent_process.set_response(
                Response(
                    response_message = f"OpenAI BAD REQUEST error {e.status_code}: (e.response)"
                )
            )
        except Exception as e:
            agent_process.set_response(
                Response(
                    response_message = f"An unexpected error occurred: {e}"
                )
            )

        agent_process.set_status("done")
        agent_process.set_end_time(time.time())
