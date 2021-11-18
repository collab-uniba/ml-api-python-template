""" Main operations with Command line interface (CLI). """

import tempfile
import warnings
from pathlib import Path
from typing import Optional

import mlflow
import typer

from config.config import MODELS_DIR, logger
from template import utils
from template.data import data
from template.models import train

# Ignore warning
warnings.filterwarnings("ignore")
# Typer CLI app
app = typer.Typer()


@app.command()
def train_model(
    dataset_fp: str = typer.Argument(..., help="Path to dataset file"),
    params_fp: Optional[str] = Path.joinpath(MODELS_DIR, "params.json").read_text(),
    experiment_name: Optional[str] = "best",
    run_name: Optional[str] = "model",
) -> None:
    """Train a model using the specified parameters.

    Args:
        params_fp (Path, optional): Parameters to use for training.
                                    Defaults to `config/params.json`.
        experiment_name (str, optional): Name of the experiment to save the run to.
                                         Defaults to `best`.
        run_name (str, optional): Name of the run. Defaults to `model`.

    Returns:
        None
    """

    artifacts = {}

    logger.info("Starting MLflow experiment run")
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        logger.info("Run ID: %s", run_id)

        # Data preparation
        logger.info("Preparing data...")
        raw = data.load_raw_dataset(dataset_fp)
        artifacts["dataset"] = Path(dataset_fp).stem  # get the filename without extension
        preprocessed = data.clean_dataset(raw, artifacts["dataset"])
        artifacts["features"] = data.get_features(preprocessed)

        # Parameters
        logger.info("Loading parameters...")
        artifacts["params"] = utils.load_params(params_fp)
        mlflow.log_params(artifacts["params"])

        # Set tags
        tags = {}
        mlflow.set_tags(tags)

        # Train
        logger.info("Preparing train/test datasets...")
        train_test = train.split_dataset(preprocessed)
        logger.info("Training model...")
        artifacts["model"], artifacts["metrics"] = train.train(
            dataset=train_test, params=artifacts["params"], dataset_name=artifacts["dataset"]
        )

        logger.info("Logging metrics and artifact to MLflow registry")
        # Log metrics
        mlflow.log_metrics(artifacts["metrics"])
        # Log artifacts
        with tempfile.TemporaryDirectory() as tempdir:
            utils.save_json(artifacts["params"], Path(tempdir, "params.json"))
            utils.save_json(artifacts["metrics"], Path(tempdir, "performance.json"))
            utils.save_json(artifacts["features"], Path(tempdir, "features.json"))
            utils.save_joblib(artifacts["model"], Path(tempdir, "model.joblib"))
            mlflow.log_artifacts(tempdir)

        logger.info("Logging model to MLflow registry")
        mlflow.sklearn.log_model(artifacts["model"], "HistGradientBoostingRegressor")


def run():
    """Run the CLI."""
    app()


if __name__ == "__main__":
    run()
