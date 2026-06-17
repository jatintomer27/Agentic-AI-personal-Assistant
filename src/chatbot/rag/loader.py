"""
Handles loading and extracting text from PDF and TXT files.
Supports:
    - PDF files via pdfplumber
    - TXT files via plain Python
"""

import os
from typing import Union
import pdfplumber
from pathlib import Path
from dataclasses import dataclass
from chatbot import logger


@dataclass
class Document:
    """
    Represents a loaded document.

    Attributes
    ----------
    content     : full extracted text
    filename    : original filename
    file_type   : "pdf" or "txt"
    metadata    : extra info (page count, file size etc.)
    """
    content  : str
    filename : str
    file_type: str
    metadata : dict


def load_pdf(file_path: str | Path) -> Document:
    """
    Extracts text from a PDF file page by page.

    Args:
        file_path : path to the PDF file

    Returns:
        Document object with full extracted text
    
    Raises:
        ValueError        : empty file or no text extracted
        Exception         : for any unexpected error during extraction
    """
    file_path = Path(file_path)

    # ── Validate not empty ─────────────────────────────────────────────
    if file_path.stat().st_size == 0:
        error_msg = f"PDF file is empty: {file_path.name}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    full_text = []
    metadata  = {}
    failed_pages = []

    try:
        with pdfplumber.open(file_path) as pdf:
            metadata["page_count"] = len(pdf.pages)

            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    text = page.extract_text()
                    if text and text.strip():
                        # tag each page so we know where content came from
                        full_text.append(f"[Page {page_num}]\n{text}")
                    else:
                        failed_pages.append(page_num)
                        logger.warning(
                                f"No text on page {page_num} of '{file_path.name}' "
                                f"— may be an image-based page."
                            )
                except Exception as e:
                    failed_pages.append(page_num)
                    logger.warning(
                            f"Failed to extract page {page_num} "
                            f"of '{file_path.name}': {e}"
                    )
                    continue    # skip bad page, continue with rest
    except ValueError:
        raise
    except Exception as e:
        logger.exception(
            f"Unexpected error opening PDF '{file_path.name}': {e}"
        )
        raise
    if not full_text:
        logger.error(
            f"No text extracted from '{file_path.name}'. "
            f"All {metadata.get('page_count', 0)} pages failed. "
            f"PDF may be image-based — consider using OCR."
        )
        raise ValueError(
            f"No text could be extracted from '{file_path.name}'.\n"
            f"The PDF may be image-based or scanned.\n"
            f"Consider using an OCR tool to convert it first."
        )
    total_pages   = metadata["page_count"]
    success_pages = total_pages - len(failed_pages)
    metadata = {
        **metadata,
        "file_size": os.path.getsize(file_path),
        "file_path": str(file_path),
    }
    if success_pages:
        metadata.update({"success_pages":success_pages})
    if failed_pages:
        metadata.update({"failed_pages":failed_pages})
    return Document(
        content   = "\n\n".join(full_text),
        filename  = file_path.name,
        file_type = "pdf",
        metadata  = metadata  
    )


def load_txt(file_path: str | Path) -> Document:
    """
    Loads text from a plain .txt file.

    Args:
        file_path : path to the TXT file

    Returns:
        Document object with file content
    
    Raises:
        ValueError        : file is empty
        Exception         : for any unexpected read error
    """
    file_path = Path(file_path)
    if file_path.stat().st_size == 0:
        error_msg = f"TXT file is empty: {file_path.name}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except PermissionError:
        error_msg = f"Permission denied reading file: {file_path.name}"
        logger.error(error_msg)
        raise PermissionError(error_msg)
    except Exception as e:
        logger.exception(
            f"Unexpected error reading '{file_path.name}': {e}"
        )
        raise
    
    if not content.strip():
        logger.error(f"TXT file has no readable content: {file_path.name}")
        raise ValueError(
            f"'{file_path.name}' contains no readable text."
        )

    logger.info(
        f"'{file_path.name}' loaded successfully — "
        f"{len(content)} characters extracted."
    )

    return Document(
        content   = content,
        filename  = file_path.name,
        file_type = "txt",
        metadata  = {
            "file_size": os.path.getsize(file_path),
            "file_path": str(file_path),
        }
    )


def load_file(file_path: str | Path) -> Document:
    """
    Auto-detects file type and loads accordingly.
    Supports .pdf and .txt files.

    Args:
        file_path : path to the file

    Returns:
        Document object

    Raises:
        FileNotFoundError : if file does not exist
        ValueError        : if file type is not supported
    """
    file_path = Path(file_path)

    # Validate file exist
    if not file_path.exists():
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(file_path)

    elif extension == ".txt":
        return load_txt(file_path)
    else:
        error_msg = (
            f"Unsupported file type: '{extension}' for file '{file_path.name}'. "
            f"Supported types: .pdf, .txt"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


def save_uploaded_file(uploaded_file, uploads_dir: str | Path) -> Path:
    """
    Saves a Streamlit uploaded file object to disk.

    Args:
        uploaded_file : Streamlit UploadedFile object
        uploads_dir   : directory to save the file

    Returns:
        Path to the saved file

    Raises:
        ValueError      : if uploaded_file is None or has no content
        PermissionError : if directory cannot be created or written to
        Exception       : for any unexpected error
    """
    if uploaded_file is None:
        logger.error("uploaded_file is None — no file was provided.")
        raise ValueError("No file provided to save.")

    if uploaded_file.size == 0:
        logger.error(f"Uploaded file '{uploaded_file.name}' is empty.")
        raise ValueError(
            f"Uploaded file '{uploaded_file.name}' is empty."
        )

    uploads_dir = Path(uploads_dir)

    try:
        uploads_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        logger.error(f"Permission denied creating directory: {uploads_dir}")
        raise PermissionError(
            f"Cannot create uploads directory: {uploads_dir}\n"
            f"Check folder permissions."
        )

    file_path = uploads_dir / uploaded_file.name

    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    except PermissionError:
        logger.error(f"Permission denied writing file: {file_path}")
        raise PermissionError(
            f"Cannot write file '{uploaded_file.name}' to '{uploads_dir}'.\n"
            f"Check folder permissions."
        )
    except Exception as e:
        logger.exception(
            f"Unexpected error saving '{uploaded_file.name}': {e}"
        )
        raise
    logger.info(f"{uploaded_file} file is saved successfully at {file_path}")
    return file_path


def get_uploaded_files(uploads_dir: str | Path) -> list[Path]:
    """
    Returns all PDF and TXT files in the uploads directory.

    Args:
        uploads_dir : directory to scan

    Returns:
        List of file paths (empty list if directory doesn't exist)

    Raises:
        PermissionError : if directory cannot be read
        Exception       : for any unexpected error
    """
    uploads_dir = Path(uploads_dir)
    if not uploads_dir.exists():
        logger.warning(
            f"Uploads directory does not exist: '{uploads_dir}' "
            f"— returning empty list."
        )
        return []
    try:
        list_of_file_path = [
            f for f in uploads_dir.iterdir()
            if f.suffix.lower() in (".pdf", ".txt")
        ]
    except PermissionError:
        logger.error(f"Permission denied reading directory: '{uploads_dir}'")
        raise PermissionError(
            f"Cannot read uploads directory: '{uploads_dir}'.\n"
            f"Check folder permissions."
        )

    except Exception as e:
        logger.exception(
            f"Unexpected error scanning '{uploads_dir}': {e}"
        )
        raise

    logger.info(
        f"{len(list_of_file_path)} file(s) found in '{uploads_dir}': "
        f"{[f.name for f in list_of_file_path]}"
    )
    return list_of_file_path
