"""
Train a scikit-learn model on UCI Wine Quality Dataset
https://archive.ics.uci.edu/ml/datasets/wine+quality
"""

from pathlib import Path

import pandas as pd

from config.config import DATA_DIR, logger


def preprocess(raw_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw dataset.
    """
    logger.info("Preprocessing the raw dataset")
    logger.info("Dataframe head")
    print(raw_dataset.head())
    logger.info("Dataframe description")
    print(raw_dataset.describe())

    # remove duplicate and rows with missing values
    dataset = raw_dataset.drop_duplicates().dropna()
    # remove colums with low variance
    dataset = dataset.loc[:, dataset.std() > 0.1]
    logger.info("Updated datframe description")
    print(dataset.describe())

    dataset.to_csv(
        Path.joinpath(DATA_DIR, "processed", "wine_quality_preprocessed.csv"), index=False
    )

    logger.info(
        "Preprocessed dataset saved to: %s",
        Path.joinpath(DATA_DIR, "processed", "wine_quality_preprocessed.csv"),
    )


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load the raw dataset.
    """
    dataset = pd.read_csv(path, delimiter=";")
    dataset = dataset.rename(columns=lambda x: x.lower().replace(" ", "_"))
    return dataset


def run():
    """
    Main function.
    """
    _path = Path.joinpath(DATA_DIR, "raw", "winequality-red.csv")
    raw = load_raw_data(_path)
    preprocess(raw)


if __name__ == "__main__":
    run()
