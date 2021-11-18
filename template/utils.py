import json
import pathlib
from typing import Dict

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
        with open(path, "w") as f:
            json.dump(data, f, indent=4, sort_keys=True)
    except Exception as e:
        logger.error(e)


def save_joblib(data, path):
    """Save data to joblib file

    Args:
        data (object): The data to be saved
        path (pathlib.Path): path to save

    Returns:
        None
    """
    try:
        import joblib

        joblib.dump(data, path)
    except Exception as e:
        logger.error(e)


def load_params(path: str) -> json:
    """Load params from json file

    Args:
        path (str): The file path to load

    Returns:
        params (json): loaded params
    """
    return json.loads(path)
