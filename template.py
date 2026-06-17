"""
File for ceating project template structure.

Execute that file and create the complete folder structure.
"""

import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]:%(message)s:')


PROJECT_NAME = 'chatbot'


# List files and folder want to create inside directory

list_of_files = [
    f"src/{PROJECT_NAME}/__init__.py",
    f"src/{PROJECT_NAME}/graph/__init__.py",
    f"src/{PROJECT_NAME}/graph/state.py",
    f"src/{PROJECT_NAME}/graph/nodes.py",
    f"src/{PROJECT_NAME}/graph/builder.py",
    f"src/{PROJECT_NAME}/graph/llm_factory.py",


    f"src/{PROJECT_NAME}/database/__init__.py",
    f"src/{PROJECT_NAME}/database/models.py",
    f"src/{PROJECT_NAME}/database/connection.py",
    f"src/{PROJECT_NAME}/database/repository.py",


    f"src/{PROJECT_NAME}/tools/__init__.py",
    f"src/{PROJECT_NAME}/tools/calculator.py",

    f"src/{PROJECT_NAME}/utils/__init__.py",
    f"src/{PROJECT_NAME}/utils/common.py",

    f"src/{PROJECT_NAME}/constants/__init__.py",


    f"src/{PROJECT_NAME}/rag/__init__.py",
    f"src/{PROJECT_NAME}/rag/loader.py",
    f"src/{PROJECT_NAME}/rag/splitter.py",
    f"src/{PROJECT_NAME}/rag/vector_store.py",
    f"src/{PROJECT_NAME}/rag/embedding_llm_factory.py",
    f"src/{PROJECT_NAME}/rag/pipeline.py",

    "config/settings.yaml",
    "app.py",
    "requirements.txt",
    'pyproject.toml',
]


for file in list_of_files:
    filepath = Path(file)
    filedir , filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath,"w") as f:
            pass
            logging.info(f"Creating empty file: {filename}")
    else:
        logging.info(f"{filename} is already exists")