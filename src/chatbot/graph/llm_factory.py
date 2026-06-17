"""
Returns a LangChain-compatible LLM instance based on config/config.yaml.
To switch providers, just change LLM_PROVIDER in config — nothing else changes.
"""

from __future__ import annotations
import os
from box import ConfigBox
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from chatbot import logger


load_dotenv()

def get_llm(config: ConfigBox) -> BaseChatModel:
    """
    Factory function — returns the configured LLM.

    Supported providers:
        "anthropic"     → Claude models via langchain-anthropic
        "openai"        → GPT models via langchain-openai
        "groq"          → Groq-hosted models via langchain-groq
        "google_gemini" → Google’s Gemini models via langchain-google-genai
    """
    provider = config.llm.provider
    model = config.llm.model
    temperature = config.llm.temperature or 0.5
    max_tokens = config.llm.max_tokens or 2048

    api_key = os.getenv('LLM_API_KEY')
    if not api_key:
        raise ValueError(
            f"LLM_API_KEY is missing in your .env file.\n"
            f"Provider '{provider}' requires an API key.\n"
            f"Fix: Add 'LLM_API_KEY=your_key_here' to your .env file."
        )
    logger.info(f"Loading embedding model — provider: '{provider}', model: '{model}'")
    try:
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic
            llm_model = ChatAnthropic(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )

        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            llm_model = ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )
        elif provider == "groq":
            from langchain_groq import ChatGroq
            llm_model = ChatGroq(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )
        elif provider == 'google_gemini':
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm_model = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )
        else:
            raise ValueError(
                f"Unknown LLM_PROVIDER: '{provider}'. "
                "Choose from: 'anthropic', 'openai', 'groq', 'google_gemini'"
            )
        logger.info(
            f"LLM model loaded successfully — "
            f"provider: '{provider}', model: '{model}'"
        )
        return llm_model
    except ValueError:
        raise
    except ImportError as e:
        packages = {
            "openai"      : "langchain-openai",
            "anthropic"   : "langchain-anthropic",
            "groq" : "langchain-groq",
            "google_gemini": "langchain_google_genai",
        }
        install = packages.get(provider, "the required package")
        logger.error(
            f"Missing package for provider '{provider}': {e}\n"
            f"Fix: pip install {install}"
        )
        raise ImportError(
            f"Package not installed for provider '{provider}'.\n"
            f"Run: pip install {install}"
        ) from e
    except Exception as e:
        logger.exception(
            f"Unexpected error loading LLM model "
            f"(provider='{provider}', model='{model}'): {e}"
        )
        raise

    
