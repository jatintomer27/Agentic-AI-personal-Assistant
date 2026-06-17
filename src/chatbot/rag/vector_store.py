"""
Handles storing chunks as vectors in ChromaDB and retrieving
relevant chunks for a given query.

ChromaDB runs locally — no external service needed.
Data is persisted to disk at chroma_db/ directory.
"""

import os
from typing import Union
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions

from chatbot import logger
from chatbot.rag.splitter import Chunk
from chatbot.rag.embedding_llm_factory import get_embedding_model

load_dotenv()

# ─────────────────────────────────────────────
#  Build ChromaDB Local client
# ─────────────────────────────────────────────

def get_chroma_local_client(
        vector_store_dir: str
) -> chromadb.PersistentClient:
    """
    Creates a persistent ChromaDB client.
    Data is saved to disk at vector_store_dir.

    Args:
        vector_store_dir : path to store ChromaDB data

    Returns:
        ChromaDB PersistentClient
    """
    return chromadb.PersistentClient(path=vector_store_dir)

# ─────────────────────────────────────────────
#  Build ChromaDB Cloud client
# ─────────────────────────────────────────────

def get_chroma_cloud_client(
    api_key: str,
    tenant: str,
    database: str,
) -> chromadb.CloudClient:
    """
    Create and return a Chroma Cloud client.

    Args:
        api_key (str):
            Chroma Cloud API key used for authentication.
        tenant (str):
            Chroma Cloud tenant identifier.
        database (str):
            Name of the Chroma Cloud database to connect to.

    Returns:
        chromadb.CloudClient:
            A configured Chroma Cloud client instance that can be used
            to create, query, and manage collections.
    """
    return chromadb.CloudClient(
        api_key=api_key,
        tenant=tenant,
        database=database,
    )

# ─────────────────────────────────────────────
#  Build ChromaDB client
# ─────────────────────────────────────────────

def get_chroma_client(config):
    """
    Factory — returns local or cloud ChromaDB client
    based on vector_db.usage value in settings.yaml file
    """
    usage = config.rag.vector_db.usage.lower()  # "local" or "cloud"

    try:
        if usage == "local":
            return get_chroma_local_client(
                config.rag.vector_db.local.persist_directory
            )

        elif usage == "cloud":
            error_message = ""
            if not os.getenv("CHROMA_API_KEY"):
                error_message = "CHROMA_API_KEY is not set in .env file. "
            if not os.getenv("CHROMA_TENANT"):
                error_message += "CHROMA_TENANT is not set in .env file. "
            if error_message:
                logger.error(error_message)
                raise ValueError(error_message)
            return get_chroma_cloud_client(
                api_key  = os.getenv("CHROMA_API_KEY"),
                tenant   = os.getenv("CHROMA_TENANT"),
                database = config.rag.vector_db.cloud.database,
            )

        else:
            error_message = (
                f"Unknown vector_db.usage: '{usage}'. "
                "Choose from: 'local' or 'cloud'"
            )
            logger.error(error_message)
            raise ValueError(error_message)
    except ValueError:
        raise
    except Exception as e:
        error_message = (
                f"An unexpected error occured while get the chroma client. {e}"
            )
        logger.exception(error_message)
        raise

# ─────────────────────────────────────────────
#  Get or Create ChromaDB collections
# ─────────────────────────────────────────────

def get_collection(
    client          : Union[chromadb.PersistentClient, chromadb.CloudClient],
    collection_name : str,
    embedding_provider: str,
    embedding_model : str,
):
    """
    Gets or creates a ChromaDB collection.
    A collection is like a table — stores vectors + metadata.

    Args:
        client              : ChromaDB client
        collection_name     : name for the collection
        embedding_provider  : Provider of embedding model
        embedding_model     : Embedding model use to do embedding

    Returns:
        ChromaDB Collection
    """
    # Get embeddings model according to the setting.yaml setting
    try:
        embedding_fn = get_embedding_model(embedding_provider, embedding_model)
    except Exception as e:
         raise e

    try:
        # get_or_create — safe to call every time
        collection = client.get_or_create_collection(
            name               = collection_name,
            embedding_function = embedding_fn,
            metadata           = {"hnsw:space": "cosine"},  # cosine similarity
        )
    except Exception as e:
        logger.error(
            f"Error occured while creating the Collection —"
            f"Collection name: '{collection_name}'"
            f"with embedding function: '{embedding_fn}'"
            f"with error: {e}"
        )
        raise e

    return collection


# ─────────────────────────────────────────────
#  Store chunks
# ─────────────────────────────────────────────

def store_chunks(collection, chunks: list[Chunk], filename: str) -> int:
    """
    Stores a list of Chunk objects in ChromaDB.
    Skips if this file's chunks are already stored.

    Args:
        collection : ChromaDB collection
        chunks     : list of Chunk objects from splitter.py
        filename   : source filename (used to check duplicates)

    Returns:
        Number of chunks stored (0 if already existed)
    """
    # check if this file was already processed
    existing = collection.get(where={"filename": filename})
    if existing["ids"]:
        logger.info(f"{filename}' already in vector store — skipping.") 
        return 0

    # prepare data for ChromaDB
    ids       = [f"{filename}_chunk_{chunk.chunk_index}" for chunk in chunks]
    documents = [chunk.content  for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]

    # store in ChromaDB — embeddings generated automatically
    collection.add(
        ids       = ids,
        documents = documents,
        metadatas = metadatas,
    )
    logger.info(f"Stored {len(chunks)} chunks from '{filename}'")
    return len(chunks)


def delete_file_chunks(collection, filename: str) -> bool:
    """
    Deletes all chunks for a specific file from the vector store.
    Called when user removes a file.

    Args:
        collection : ChromaDB collection
        filename   : filename to delete

    Returns:
        True if deleted, False if file not found
    """
    try:
        existing = collection.get(where={"filename": filename})
    except Exception as e:
        logger.error(f"Failed to get existing files collection from ChromaDB: {e}")
        raise

    if not existing["ids"]:
        logger.info(f"File {filename} is not found in collection {collection}")
        return False

    try:
        collection.delete(where={"filename": filename})
        logger.info(f"Deleted chunks for '{filename}' in collection {collection}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete the file from collection of ChromaDB: {e}")
        raise

def get_stored_files(collection) -> list[str]:
    """
    Returns list of unique filenames already in the vector store.
    Used to show which files have been processed.

    Returns:
        List of filenames
    """
    
    try:
        results = collection.get()
    except Exception as e:
        logger.error(f"Failed to get collection from ChromaDB: {e}")
        raise

    if not results["metadatas"]:
        logger.info(f"No file found in the collection {collection}")
        return []

    try:
        # extract unique filenames from metadata
        filenames = list(set(
            meta["filename"]
            for meta in results["metadatas"]
            if "filename" in meta
        ))
    except Exception as e:
        logger.error(f"Failed to process retireved file names from ChromaDB.")
        raise
    logger.info(f"{filenames} files names are fetched from the collection {collection}")
    return filenames


# ─────────────────────────────────────────────
#  Retrieve relevant chunks
# ─────────────────────────────────────────────

def retrieve_relevant_chunks(
    collection,
    query  : str,
    top_k  : int = 3,
) -> str:
    """
    Searches ChromaDB for chunks most relevant to the query.
    Returns them as a single formatted string to inject into the prompt.

    Args:
        collection : ChromaDB collection
        query      : user's question
        top_k      : number of chunks to retrieve

    Returns:
        Formatted string of relevant context
    """
    if not isinstance(query, str) or not query.strip():
        raise TypeError(
            f"query must be a non-empty string, got '{query}'"
        )

    if not isinstance(top_k, int):
        raise TypeError(
            f"top_k must be an int, got {type(top_k).__name__}"
        )

    if top_k <= 0:
        raise ValueError(
            f"top_k must be > 0, got {top_k}"
        )

    try:
        count = collection.count()
    except Exception as e:
        logger.error(f"Failed to get collection count from ChromaDB: {e}")
        raise

    # check if collection has any data
    if count == 0:
        return ""

    # query ChromaDB — it embeds the query and finds similar chunks
    try:
        results = collection.query(
            query_texts = [query],
            n_results   = min(top_k, collection.count()),
        )
    except Exception as e:
        logger.error(
            f"ChromaDB query failed for query '{query[:60]}...': {e}"
        )
        raise

    if not results["documents"] or not results["documents"][0]:
        logger.info(f"No relevant chunks found for query: '{query[:60]}...'")
        return ""

    try:
        # format retrieved chunks into readable context
        context_parts = []
        for i, (doc, meta) in enumerate(
            zip(results["documents"][0], results["metadatas"][0]), start=1
        ):
            source = meta.get("filename", "unknown")
            context_parts.append(f"[Source {i}: {source}]\n{doc}")
        context = "\n\n".join(context_parts)
    except Exception as e:
        logger.error(
            f"Failed to format retrieved chunks for query '{query[:60]}...': {e}"
        )
        raise
    logger.info(
        f"Retrieved {len(context_parts)} chunk(s) for query: '{query[:60]}...'"
    )
    return context
