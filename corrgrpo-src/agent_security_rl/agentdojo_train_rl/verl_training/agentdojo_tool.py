"""VERL tool adapter that executes official AgentDojo functions."""

from __future__ import annotations

import json
from ast import literal_eval
from typing import Any

import yaml
from agentdojo.functions_runtime import FunctionCall
from pydantic import BaseModel
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

from verl_training.state import STATE_KEY, AgentDojoTrajectoryState


def _is_string_list(value: str) -> bool:
    try:
        return isinstance(literal_eval(value), list)
    except (ValueError, SyntaxError):
        return False


def _tool_result_to_str(result: Any) -> str:
    """Match AgentDojo's YAML tool-output format without importing provider SDKs."""
    if isinstance(result, BaseModel):
        result = result.model_dump()
    elif isinstance(result, list):
        converted = []
        for item in result:
            if type(item) in (str, int):
                converted.append(str(item))
            elif isinstance(item, BaseModel):
                converted.append(item.model_dump())
            else:
                raise TypeError(f"Unsupported tool result item: {type(item).__name__}")
        result = converted
    else:
        return str(result)
    return yaml.safe_dump(result, allow_unicode=True).strip()


class AgentDojoTool(BaseTool):
    """Dispatch one OpenAI tool name into the trajectory's AgentDojo runtime."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        # BaseTool prints every schema at construction. There are 69 AgentDojo
        # tools, so initialize the same fields directly to keep worker logs sane.
        if tool_schema is None:
            raise ValueError("AgentDojoTool requires an explicit tool_schema")
        self.config = config
        self.tool_schema = tool_schema
        self.name = tool_schema.function.name
        configured_name = config.get("function_name")
        if configured_name and configured_name != self.name:
            raise ValueError(f"Tool config name {configured_name!r} != schema name {self.name!r}")

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        *,
        agent_data: Any,
        **_: Any,
    ) -> tuple[ToolResponse, float, dict[str, Any]]:
        del instance_id
        state = agent_data.extra_fields.get(STATE_KEY)
        if not isinstance(state, AgentDojoTrajectoryState):
            return ToolResponse(text="AgentDojo environment was not initialized."), 0.0, {"valid": 0.0}

        normalized_parameters = dict(parameters)
        for key, value in normalized_parameters.items():
            if isinstance(value, str) and _is_string_list(value):
                normalized_parameters[key] = literal_eval(value)

        call = FunctionCall(function=self.name, args=normalized_parameters)
        state.function_calls.append(call)
        result, error = state.runtime.run_function(
            state.environment,
            self.name,
            normalized_parameters,
        )
        if error is not None:
            text = json.dumps({"error": error}, ensure_ascii=False)
            return ToolResponse(text=text), 0.0, {"tool_error": 1.0}

        text = _tool_result_to_str(result)
        return ToolResponse(text=text), 0.0, {"tool_error": 0.0}
