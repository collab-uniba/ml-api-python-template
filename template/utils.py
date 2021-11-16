import json

from config.config import logger


def save_json(data, path):
    """
    Save data to json file
    :param data: data to be saved
    :param path: path to save
    :return: None
    """
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4, sort_keys=True)
    except Exception as e:
        logger.error(e)
