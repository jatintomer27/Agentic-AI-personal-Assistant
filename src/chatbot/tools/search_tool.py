"""
Module contains the search tool.
"""
from langchain_core.tools import tool
from chatbot.rag.pipeline import RAGPipeline
from chatbot.utils.common import load_config_file

try:
    config = load_config_file(__file__)
except Exception as e:
    raise

@tool
def search_documents(query: str) -> str:
    """
    Search the uploaded documents for information relevant to the query.
    Use this when the user asks about topics that may be in their uploaded files.
    """
    rag = RAGPipeline(config)
    content = rag.retrieve(query)
    return {
        'query':query,
        'content':content
    }