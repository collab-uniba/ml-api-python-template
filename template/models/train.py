"""[summary]

Returns:
    [type]: [description]
"""

from pathlib import Path
from typing import Dict

import pandas as pd
from joblib import dump
from sklearn import preprocessing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from config.config import BASE_DIR, DATA_DIR, logger


def prepare_dataset(
    dataset: pd.DataFrame, test_size: int = 0.2, random_seed: int = 1234
) -> Dict[str, pd.DataFrame]:
    """[summary]

    Args:
        dataset (pd.DataFrame): [description]
        test_size (int, optional): [description]. Defaults to 0.2.
        random_seed (int, optional): [description]. Defaults to 1234.

    Returns:
        Dict[str, pd.DataFrame]: [description]
    """
    logger.info("Splliting dataset...")
    train_df, test_df = train_test_split(dataset, test_size=test_size, random_state=random_seed)
    return {"train": train_df, "test": test_df}


def train(dataset: Dict) -> None:
    """
    Train a model and save it to disk
    """
    train_df = dataset["train"]
    test_df = dataset["test"]

    # separate features from target
    y_train = train_df["quality"]
    x_train = train_df.drop("quality", axis=1)
    y_test = test_df["quality"]
    x_test = test_df.drop("quality", axis=1)

    logger.info("Training model...")
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    model = HistGradientBoostingRegressor(max_iter=50).fit(x_train, y_train)

    y_pred = model.predict(x_test)
    error = mean_squared_error(y_test, y_pred)
    logger.info(f"Test MSE: {error}")

    logger.info("Saving artifacts...")
    _path = Path(Path.joinpath(BASE_DIR, "artifacts"))
    _path.mkdir(exist_ok=True)
    dump(model, Path.joinpath(_path, "model.joblib"))

    logger.info("Done!")


def run():
    """
    Run the training process
    """
    logger.info("Loading dataset...")
    dataset = pd.read_csv(Path.joinpath(DATA_DIR, "processed", "wine_quality_preprocessed.csv"))
    train_test = prepare_dataset(dataset)
    train(train_test)


if __name__ == "__main__":
    run()
