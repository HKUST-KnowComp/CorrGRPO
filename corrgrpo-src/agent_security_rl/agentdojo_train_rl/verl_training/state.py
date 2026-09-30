"""Per-trajectory AgentDojo state shared by the agent loop and tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime, TaskEnvironment


STATE_KEY = "_agentdojo_private_state"


@dataclass
class AgentDojoTrajectoryState:
    suite: Any
    user_task: Any
    injection_task: Any | None
    runtime: FunctionsRuntime
    pre_environment: TaskEnvironment
    environment: TaskEnvironment
    function_calls: list[FunctionCall] = field(default_factory=list)
    final_model_output: str = ""
    format_valid: bool = True
