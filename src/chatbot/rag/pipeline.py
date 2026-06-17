"""
Orchestrates the full RAG workflow.
This is the only file the rest of the app talks to for RAG operations.

Two main workflows:
    1. ingest_file()    → upload + process + store a file
    2. retrieve()       → search and return relevant context for a query
"""

import os
from pathlib import Path

from chatbot import logger
from chatbot.rag.loader import load_file, save_uploaded_file, get_uploaded_files
from chatbot.rag.splitter import split_document
from chatbot.rag.vector_store import (
    get_chroma_client,
    get_collection,
    store_chunks,
    delete_file_chunks,
    get_stored_files,
    retrieve_relevant_chunks,
)


class RAGPipeline:
    """
    Single entry point for all RAG operations.

    Usage:
        rag = RAGPipeline(config)

        # ingest a file
        rag.ingest_file(uploaded_file)

        # retrieve context for a query
        context = rag.retrieve("What is machine learning?")
    """

    def __init__(self, config):
        """
        Initializes the RAG pipeline with config from settings.yaml.

        Args:
            config : the rag section of your settings.yaml
                     config.rag.uploads_dir
                     config.rag.chunk_size
                     config.rag.chunk_overlap
                     config.rag.top_k
        """
        try:
            self.uploads_dir          = config.rag.uploads_dir
            self.chunk_size           = config.rag.chunk_size
            self.chunk_overlap        = config.rag.chunk_overlap
            self.top_k                = config.rag.top_k

            # initialize ChromaDB
            logger.info(
                f"Initializing RAG pipeline — "
                f"usage: '{config.rag.vector_db.usage}', "
                f"uploads_dir: '{self.uploads_dir}'"
            )

            self.client     = get_chroma_client(config)
            self.collection = get_collection(
                client          = self.client,
                collection_name = config.rag.vector_db.collection_name,
                embedding_provider = config.rag.embedding.provider,
                embedding_model = config.rag.embedding.model,
            )
            logger.info("RAG pipeline initialized successfully.")
        except ValueError as e:
            logger.error(f"RAG pipeline configuration error: {e}")
            raise
        except Exception as e:
            # anything unexpected — connection refused, wrong credentials etc.
            logger.exception(
                f"Failed to initialize RAG pipeline: {e}\n"
                f"Check your vector_db settings in settings.yaml and .env file."
            )
            raise

    # ─────────────────────────────────────────
    #  Ingest workflow
    # ─────────────────────────────────────────

    def ingest_file(self, uploaded_file) -> dict:
        """
        Full ingestion pipeline for a Streamlit uploaded file.

        Steps:
            1. Save file to disk
            2. Extract text
            3. Split into chunks
            4. Store in ChromaDB

        Args:
            uploaded_file : Streamlit UploadedFile object

        Returns:
            dict with status and chunk count
        """
        try:
            # Step 1 — save to disk
            file_path = save_uploaded_file(uploaded_file, self.uploads_dir)

            # Step 2 — extract text
            document = load_file(file_path)

            
            # Step 3 — split into chunks
            chunks = split_document(document, self.chunk_size, self.chunk_overlap)

            # Step 4 — store in ChromaDB
            stored = store_chunks(self.collection, chunks, document.filename)

            return {
                "status"     : "success",
                "filename"   : document.filename,
                "chunks"     : len(chunks),
                "stored"     : stored,
                "already_existed": stored == 0,
            }
        except Exception as e:
            logger.error(f"Error occured while file ingestion: {e}")
            return {
                "status" : "error",
                "error"  : str(e),
            }

    def ingest_existing_files(self) -> list[dict]:
        """
        Processes all files already saved in the uploads directory.
        Useful on app startup to ensure all uploaded files are indexed.

        Returns:
            List of result dicts from ingest_file
        """
        
        try:
            files  = get_uploaded_files(self.uploads_dir)
        except Exception as e:
            raise
        try:
            stored_files = get_stored_files(self.collection)
        except Exception as e:
            logger.exception(f"Unexpected error fetching stored files from vector store: {e}")
            raise

        results = []
        for file_path in files:
            
            # check if already in vector store
            if file_path.name in stored_files:
                logger.info(f"'{file_path.name}' already indexed — skipping")
                continue

            # load and ingest
            try:
                document = load_file(file_path)
                chunks   = split_document(document, self.chunk_size, self.chunk_overlap)
                stored   = store_chunks(self.collection, chunks, document.filename)
                results.append({"status": "success", "filename": file_path.name, "chunks": stored})
            except Exception as e:
                results.append({"status": "error", "filename": file_path.name, "error": str(e)})

        return results

    # ─────────────────────────────────────────
    #  Retrieve workflow
    # ─────────────────────────────────────────

    def retrieve(self, query: str) -> str:
        """
        Retrieves relevant context chunks for a user query.

        Args:
            query : user's question

        Returns:
            Formatted context string to inject into the LLM prompt.
            Returns empty string if no relevant content found.
        """
        if not isinstance(query, str):
            raise TypeError(
                f"query must be a string, got {type(query).__name__}"
            )

        if not query.strip():
            logger.warning("Empty query received in retrieve() — returning empty context.")
            return "" 
        try:                                              
            context = retrieve_relevant_chunks(
                collection = self.collection,
                query      = query,
                top_k      = self.top_k,
            )
            if not context:
                logger.info(f"No relevant chunks found for query: '{query[:60]}...'")
                return ""
            logger.info(f"Context retrieved successfully for query: '{query[:60]}...'")
            return context 
        except Exception as e:
            logger.exception(
                f"Unexpected error during retrieval for query '{query[:60]}...': {e}"
            )
            return ""                                                                                         

    # ─────────────────────────────────────────
    #  Utility
    # ─────────────────────────────────────────

    def get_indexed_files(self) -> list[str]:
        """Returns list of filenames currently in the vector store."""
        indexed_files = []
        try:
            indexed_files = get_stored_files(self.collection)
        except Exception as e:
            logger.error(f"Error occured while fetching the files stored in Vector DB")
        return indexed_files

    def delete_file(self, filename: str) -> bool:
        """
        Removes a file's chunks from the vector store.

        Args:
            filename : name of file to remove

        Returns:
            True if deleted, False if not found
        """
        
        try:
            file_path = Path(f"{self.uploads_dir}{filename}")
            file_deleted = delete_file_chunks(self.collection, filename)
            os.remove(file_path)
            logger.info(f"{file_path} is successfully deleted.")
        except Exception as e:
            file_deleted = False
        return file_deleted
