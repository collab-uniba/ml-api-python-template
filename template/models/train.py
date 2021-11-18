""" Model training
"""

from pathlib import Path
from typing import Dict

import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

from config.config import DATA_DIR, MODELS_DIR, logger

from ..utils import load_params, save_joblib, save_json
from .eval import evaluation


def train(
    dataset: Dict[str, pd.DataFrame], params: str, dataset_name: str = "winequality"
) -> None:
    """Train a model and save it to disk

    Args:
        dataset (Dict[str, pd.DataFrame]): The train and test sets.
        params (Dict[str, object]): The parameters dictionary.
        dataset_name (str, optional): The name of the dataset. Defaults to "winequality".

    Returns:
        (object, json): The trained model and the evaluation results
    """
    train_df = dataset["train"]
    test_df = dataset["test"]

    # separate features from target
    y_train = train_df["quality"]
    x_train = train_df.drop("quality", axis=1)
    y_test = test_df["quality"]
    x_test = test_df.drop("quality", axis=1)

    logger.info("Training model...")
    scaler = preprocessing.StandardScaler().fit(x_train.values)
    x_train = scaler.transform(x_train.values)
    x_test = scaler.transform(x_test.values)

    logger.info("Parameters: %s", params)
    model = HistGradientBoostingRegressor(max_iter=params["max_iter"], loss=params["loss"]).fit(
        x_train, y_train
    )

    y_pred = model.predict(x_test)
    logger.info("Computing model evaluation metrics")
    metrics = evaluation(y_test, y_pred)

    # logger.info("Saving artifacts...")
    save_json(metrics, Path.joinpath(MODELS_DIR, f"{dataset_name}_performance.json"))
    save_joblib(model, Path.joinpath(MODELS_DIR, f"{dataset_name}_model.joblib"))
    save_joblib(scaler, Path.joinpath(MODELS_DIR, f"{dataset_name}_scaler.joblib"))

    return model, metrics


def split_dataset(
    dataset: pd.DataFrame, split_ratio: int = 0.2, random_seed: int = 1234
) -> Dict[str, pd.DataFrame]:
    """Split the dataset into train and test sets.

    Args:
        dataset (pd.DataFrame): The dataset to split
        split_ratio (int, optional): The train/test split ratio. Defaults to 0.2.
        random_seed (int, optional): The random seed to ensure replicability. Defaults to 1234.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing the train and test sets.
    """
    logger.info("Splliting the 80/20 train/test datasets...")
    train_df, test_df = train_test_split(dataset, test_size=split_ratio, random_state=random_seed)
    return {"train": train_df, "test": test_df}


def load_dataset(ds_file: str) -> pd.DataFrame:
    """
    Load the cleaned dataset

    Args:
        ds_file (str): The path to the preprocessed dataset

    Returns:
        pd.DataFrame: The loaded dataset
    """
    logger.info("Loading the dataset...")
    return pd.read_csv(ds_file)


def run(
    dataset_file: str = Path.joinpath(DATA_DIR, "processed", "winequality_clean.csv"),
    params_file: str = Path.joinpath(MODELS_DIR, "params.json").read_text(),
):
    """
    Run the training process
    """
    dataset = load_dataset(dataset_file)
    train_test = split_dataset(dataset)
    params = load_params(params_file)
    res = train(train_test, params)
    return res


if __name__ == "__main__":
    run()
