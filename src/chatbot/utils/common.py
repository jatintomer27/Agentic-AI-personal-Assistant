"""
Contains utility functions
"""

import uuid
from pathlib import Path
import yaml
from box import ConfigBox
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from chatbot import logger

@ensure_annotations
def read_yaml(path_to_file:Path) ->ConfigBox:
    """
    read the yaml file and return

    Args:
        path_to_file(str): Path of yaml file

    Raises:
        ValueError: if yaml file is empty
        e: otherwise

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_file,'r+') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"Yaml file: {path_to_file} is loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"Yaml file: {path_to_file} is empty")
    except Exception as e:
        raise e
    
def generate_session_id() -> str:
    """
    Generate a unique session ID.

    Returns:
        str: A randomly generated UUID4 string.
    """
    return str(uuid.uuid4())


def extract_text_content(content) -> str:
    """
    Extracts clean plain text from any LangChain message content format.

    Handles:
        - Plain string          → "Hello"
        - Gemini list format    → [{'type': 'text', 'text': '...'}, 'more text']
    """
    # Case 1 — already a plain string
    if isinstance(content, str):
        return content

    # Case 2 — Gemini list format
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "".join(text_parts)

    return str(content)

def make_conversation_compatible(messages: str) ->list[dict]:
    formated_messages = []
    for message in messages:
        if isinstance(message, HumanMessage):
            role = 'user'
        elif isinstance(message, AIMessage) and not message.tool_calls:
            role = 'assistant'
        else:
            continue
        content = extract_text_content(message.content)
        
        formated_messages.append({
            'role':role,
            'content':content
        })
    return formated_messages


def load_conversation(chatbot, thread_id) -> dict:
    state = chatbot.get_state(config={'configurable':{'thread_id':thread_id}})
    conversation = state.values.get('messages',[])
    conversation = make_conversation_compatible(conversation)
    return conversation

def load_config_file(load_from_file: str):
    """
    Load application configuration from CONFIG_FILE_PATH.

    Args:
        load_from_file (str):
            Path of the module (__file__) requesting the configuration.

    Raises:
        FileNotFoundError:
            If CONFIG_FILE_PATH does not exist.

    Returns:
        ConfigBox:
            Loaded configuration.
    """
    from chatbot.constants import CONFIG_FILE_PATH
    config_path = Path(CONFIG_FILE_PATH)
    try:
        if not config_path.is_file():
            logger.error(
                f"Configuration file not found: {config_path}. "
                f"Requested by: {load_from_file}"
            )
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )
        logger.info(
            f"Loading configuration from: {config_path}. "
            f"Requested by: {load_from_file}"
        )
        config = read_yaml(config_path)
        logger.info(
            f"Configuration loaded successfully. "
            f"Requested by: {load_from_file}"
        )
        return config
    except FileNotFoundError as e:
        raise
    except Exception as err:
        logger.exception(
            f"Failed to load configuration from {config_path}. "
            f"Requested by: {load_from_file}"
            f"Due to: {err}"
        )
        raise
