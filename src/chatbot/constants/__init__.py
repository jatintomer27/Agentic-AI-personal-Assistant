"""
Module containing constant file paths used across the project.
"""

from pathlib import Path

CONFIG_FILE_PATH = Path(__file__).resolve().parents[3] / "config" / "settings.yaml"
