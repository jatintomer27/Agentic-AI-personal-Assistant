from chatbot.tools.calculator import calculator
from chatbot.tools.search_tool import search_documents
from chatbot.utils.common import load_config_file
from chatbot import logger

try:
    config = load_config_file(__file__)
except Exception as e:
    raise


def load_enabled_tools(tool_config, tool_registry):
    """
    Load enabled tools from configuration.

    Args:
        tool_config: ConfigBox containing tool flags.
        tool_registry: Dict[str, BaseTool]

    Returns:
        list: Enabled tool instances.
    """
    enabled_tools = []
    for tool_name, enabled in tool_config.items():
        
        if not enabled:
            continue
        
        tool = tool_registry.get(tool_name)
        
        if tool is None:
            logger.warning(
                f"Tool '{tool_name}' is enabled but not implemented."
            )
            continue

        enabled_tools.append(tool)

    logger.info(f"Loaded {len(enabled_tools)} tools.")
    return enabled_tools


# Registry of implemented tools
TOOL_REGISTRY = {
    "calculator": calculator,
    "memory_enabled": search_documents,
}

# All tools
ALL_TOOLS = load_enabled_tools(
    config.features.tools_enabled,
    TOOL_REGISTRY,
)




