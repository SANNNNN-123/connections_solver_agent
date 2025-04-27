# specify the version of the agent
__version__ = "0.16.0"

# Import all LLM interfaces to register them
import tools
import gemini_tools
import openai_tools
import grok_tools
import workflow_manager

# Get the registry instance
llm_interface_registry = tools.llm_interface_registry

# Export the registry for easy access
__all__ = ['llm_interface_registry']

