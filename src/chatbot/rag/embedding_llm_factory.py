"""
Factory function — returns the configured embedding model.
To switch providers, just change embedding.provider in config — nothing else changes.

Supported providers:
    "openai"    → OpenAI embeddings via langchain-openai
    "anthropic" → Voyage AI embeddings via langchain-voyageai
    "google"    → Google embeddings via langchain-google-genai
    "huggingface" → HuggingFace embeddings via langchain-huggingface (free, local)
    "ollama"    → Ollama local embeddings (free, fully local)
"""

from __future__ import annotations
import os
from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from chatbot import logger

load_dotenv()


def get_embedding_model(provider: str, model: str) -> Embeddings:
    """
    Factory function — returns the configured embedding model.

    All returned models are LangChain-compatible Embeddings instances.
    They all have the same interface:
        model.embed_query("some text")           → list[float]
        model.embed_documents(["text1","text2"]) → list[list[float]]
    """
    api_key = os.getenv("EMBEDDING_API_KEY")

    providers_requiring_api_key = {"openai", "anthropic", "google_gemini"}

    if provider in providers_requiring_api_key and not api_key:
        logger.error(
            f"EMBEDDING_API_KEY is not set in .env file. "
            f"Provider '{provider}' requires an API key."
        )
        raise ValueError(
            f"EMBEDDING_API_KEY is missing in your .env file.\n"
            f"Provider '{provider}' requires an API key.\n"
            f"Fix: Add 'EMBEDDING_API_KEY=your_key_here' to your .env file."
        )

    logger.info(f"Loading embedding model — provider: '{provider}', model: '{model}'")
    try:
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            embedding_model = OpenAIEmbeddings(
                model   = model,
                api_key = api_key,
            )
        elif provider == "anthropic":
            # Anthropic doesn't have its own embedding model.
            # They recommend Voyage AI — use your Anthropic API key.
            from langchain_voyageai import VoyageAIEmbeddings
            embedding_model = VoyageAIEmbeddings(
                model   = model,
                api_key = api_key,
            )
        elif provider == "google_gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embedding_model = GoogleGenerativeAIEmbeddings(
                model          = model,
                google_api_key = api_key,
            )
        elif provider == "huggingface":
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            # from langchain_huggingface import HuggingFaceEmbeddings
            embedding_model = SentenceTransformerEmbeddingFunction(
                model_name = model,
            )
        elif provider == "ollama":
            from langchain_ollama import OllamaEmbeddings
            embedding_model = OllamaEmbeddings(
                model = model,
            )
        else:
            # unsupported provider — no point trying, raise immediately
            raise ValueError(
                f"Unsupported embedding provider: '{provider}'. "
                f"Choose from: 'openai', 'anthropic', 'google_gemini', "
                f"'huggingface', 'ollama'"
            )
        logger.info(
            f"Embedding model loaded successfully — "
            f"provider: '{provider}', model: '{model}'"
        )
        return embedding_model
    except ValueError:
        # re-raise as-is — already has a clean message
        raise
    except ImportError as e:
        # tell the user exactly what to install
        packages = {
            "openai"      : "langchain-openai",
            "anthropic"   : "langchain-voyageai",
            "google_gemini": "langchain-google-genai",
            "huggingface" : "langchain-huggingface sentence-transformers",
            "ollama"      : "langchain-ollama",
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
            f"Unexpected error loading embedding model "
            f"(provider='{provider}', model='{model}'): {e}"
        )
        raise



def embed_text(text: str) -> list[float]:
    """Embeds a single text string. Used for query embedding."""
    try:
        model = get_embedding_model()
        return model.embed_query(text)
    except Exception as e:
        logger.error(
            f"Error occured while embeding the text:\n {e}"
        )
        raise

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embeds multiple texts in one call. Used for chunk embedding."""
    try:
        model = get_embedding_model()
        return model.embed_documents(texts)
    except Exception as e:
        logger.error(
            f"Error occured while embeding the texts:\n {e}"
        )
        raise