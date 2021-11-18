"""
Train a scikit-learn model on UCI Wine Quality Dataset
https://archive.ics.uci.edu/ml/datasets/wine+quality
"""

from pathlib import Path

import pandas as pd

from config.config import DATA_DIR, MODELS_DIR, logger
from template.utils import save_json


def get_features(dataset: pd.DataFrame) -> list:
    """
    Get the features from the dataset.

    Args:
        dataset (pd.DataFrame): The dataset.

    Returns:
        features (list): The features.
    """
    return list(dataset.columns)


def clean_dataset(raw_dataset: pd.DataFrame, dataset_name: str = "dataset") -> pd.DataFrame:
    """
    Preprocess the raw dataset.

    Args:
        raw_dataset (pd.DataFrame): The raw dataset.
        dataset_name (str): The name of the dataset.

    Returns:
        clean_dataset (pd.DataFrame): The preprocessed dataset.
    """
    logger.info("Preprocessing the raw dataset")
    logger.info("Dataframe head")
    print(raw_dataset.head())  # TODO move to reports?
    logger.info("Dataframe description")
    print(raw_dataset.describe())  # TODO move to reports?

    # remove duplicate and rows with missing values
    dataset = raw_dataset.drop_duplicates().dropna()
    # remove colums with low variance
    dataset = dataset.loc[:, dataset.std() > 0.1]
    logger.info("Updated datframe description")
    print(dataset.describe())  # TODO move to reports?

    _path = Path.joinpath(DATA_DIR, "processed", f"{dataset_name}_clean.csv")
    dataset.to_csv(_path, index=False)
    logger.info(
        "Preprocessed dataset saved to: %s",
        _path,
    )

    _path = Path.joinpath(MODELS_DIR, f"{dataset_name}_features.json")
    save_json(get_features(dataset), _path)
    logger.info(
        "Features saved to: %s",
        _path,
    )

    return dataset


def load_raw_dataset(path: str) -> pd.DataFrame:
    """
    Load the raw dataset.

    Args:
        path (str): The path to the dataset.

    Returns:
        dataset (pd.DataFrame): The raw dataset.
    """
    dataset = pd.read_csv(path, delimiter=";")
    dataset = dataset.rename(columns=lambda x: x.lower().replace(" ", "_"))
    return dataset


def run():
    """
    Runs the data preprocessing.
    """
    _path = Path.joinpath(DATA_DIR, "raw", "winequality-red.csv")
    raw = load_raw_dataset(_path)
    clean_dataset(raw, "winequality")


if __name__ == "__main__":
    run()
