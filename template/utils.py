"""Utility methods for data loading and saving.
"""

import json
import pathlib

import joblib

from config.config import logger


def save_json(data: object, path: pathlib.Path):
    """Save data to json file

    Args:
        data (object): The data to be saved in serializable format
        path (pathlib.Path): path to save

    Returns:
        None
    """
    try:
        with open(path, "w", encoding="utf-8") as _file:
            json.dump(data, _file, indent=4, sort_keys=True)
    except Exception as exc:
        logger.error(exc)


def save_joblib(data, path):
    """Save data to joblib file

    Args:
        data (object): The data to be saved
        path (pathlib.Path): path to save

    Returns:
        None
    """
    try:
        joblib.dump(data, path)
    except Exception as exc:
        logger.error(exc)


def load_params(path: str) -> json:
    """Load params from json file

    Args:
        path (str): The file path to load

    Returns:
        params (json): loaded params
    """
    return json.loads(path)
