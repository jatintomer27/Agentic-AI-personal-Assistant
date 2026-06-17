"""
This module sets up logging for the Customer Churn Prediction project.
Logs are saved to a file and also printed to the console.
"""

import logging
import os
import sys
from pathlib import Path

FORMAT = "[%(asctime)s]:%(levelname)s:%(filename)s:%(message)s"
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_FILEPATH = os.path.join(LOG_DIR,"running_logs.log")

os.makedirs(LOG_DIR,exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILEPATH),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)