"""
Splits large document text into smaller overlapping chunks.

Why chunking?
    - LLMs have a context window limit
    - Smaller chunks = more precise retrieval
    - Overlap ensures no information is lost at chunk boundaries
"""

from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chatbot import logger
from chatbot.rag.loader import Document


@dataclass
class Chunk:
    """
    Represents one chunk of text from a document.

    Attributes
    ----------
    content     : the chunk text
    filename    : source document filename
    chunk_index : position of this chunk in the document
    metadata    : extra info for vector store
    """
    content    : str
    filename   : str
    chunk_index: int
    metadata   : dict


def split_document(
    document   : Document,
    chunk_size : int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Splits a Document into smaller Chunk objects.

    Args:
        document      : Document object from loader.py
        chunk_size    : max characters per chunk
        chunk_overlap : characters shared between adjacent chunks

    Returns:
        List of Chunk objects

    How RecursiveCharacterTextSplitter works:
        It tries to split on paragraphs first (\n\n),
        then sentences (\n), then words ( ), then characters.
        This keeps semantically related text together.
    """

    if not isinstance(document.content, str):
        raise TypeError(
            f"document.content must be a string, "
            f"got {type(document.content).__name__} for '{document.filename}'"
        )

    if not document.content.strip():
        logger.warning(
            f"Document '{document.filename}' has empty content — "
            f"returning empty chunk list."
        )
        return []

    if chunk_size <= 0:
        raise ValueError(
            f"chunk_size must be > 0, got {chunk_size}"
        )

    if chunk_overlap < 0:
        raise ValueError(
            f"chunk_overlap must be >= 0, got {chunk_overlap}"
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size}) — overlap can't exceed chunk size."
        )

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size    = chunk_size,
            chunk_overlap = chunk_overlap,
            separators    = ["\n\n", "\n", " ", ""],  # priority order
        )

        # split the raw text into string chunks
        text_chunks = splitter.split_text(document.content)
    except Exception as e:
        logger.error(
            f"Failed to split document '{document.filename}': {e}"
        )
        raise

    try:
        # wrap each string chunk in a Chunk dataclass with metadata
        chunks = []
        for index, text in enumerate(text_chunks):
            chunks.append(Chunk(
                content     = text,
                filename    = document.filename,
                chunk_index = index,
                metadata    = {
                    "filename"   : document.filename,
                    "file_type"  : document.file_type,
                    "chunk_index": index,
                    "total_chunks": len(text_chunks),
                    **document.metadata,
                }
            ))
    except Exception as e:
        logger.error(
            f"Failed to build Chunk objects for '{document.filename}': {e}"
        )
        raise
    logger.info(
        f"Document {document.filename} is successfully splitted with "
        f"chunk size {chunk_size} and overlap {chunk_overlap}"
    )
    return chunks
