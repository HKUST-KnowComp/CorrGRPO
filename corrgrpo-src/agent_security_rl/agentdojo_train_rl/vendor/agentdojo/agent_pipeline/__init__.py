from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.agent_pipeline.errors import AbortAgentError
from agentdojo.agent_pipeline.ground_truth_pipeline import GroundTruthPipeline
from agentdojo.agent_pipeline.llms.anthropic_llm import AnthropicLLM
from agentdojo.agent_pipeline.llms.local_llm import LocalLLM
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM, OpenAILLMToolFilter
from agentdojo.agent_pipeline.llms.prompting_llm import BasePromptingLLM, PromptingLLM
from agentdojo.agent_pipeline.pi_detector import PromptInjectionDetector, TransformersBasedPIDetector
from agentdojo.agent_pipeline.planner import ToolSelector, ToolUsagePlanner
from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop, ToolsExecutor

__all__ = [
    "AbortAgentError",
    "AgentPipeline",
    "AnthropicLLM",
    "BasePipelineElement",
    "BasePromptingLLM",
    "CohereLLM",
    "GoogleLLM",
    "GroundTruthPipeline",
    "InitQuery",
    "LocalLLM",
    "OpenAILLM",
    "OpenAILLMToolFilter",
    "PipelineConfig",
    "PromptInjectionDetector",
    "PromptingLLM",
    "SystemMessage",
    "ToolSelector",
    "ToolUsagePlanner",
    "ToolsExecutionLoop",
    "ToolsExecutor",
    "TransformersBasedPIDetector",
]


def __getattr__(name):
    """Keep optional provider exports available without loading their SDKs eagerly."""
    modules = {"CohereLLM": "cohere_llm", "GoogleLLM": "google_llm"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(f"agentdojo.agent_pipeline.llms.{modules[name]}"), name)
    globals()[name] = value
    return value
