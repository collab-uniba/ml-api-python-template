""" Configuration file for the application.
"""

import configparser
import logging
import logging.config
import os
from pathlib import Path

import mlflow
import pretty_errors
from rich.logging import RichHandler

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
LOGS_DIR = Path(BASE_DIR, "logs")
DATA_DIR = Path(BASE_DIR, "data")
MODELS_DIR = Path(BASE_DIR, "models")
# STORES_DIR = Path(BASE_DIR, "stores")

# Local stores
# MODEL_REGISTRY = Path(STORES_DIR, "model")

# Create dirs
LOGS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
# MODEL_REGISTRY.mkdir(parents=True, exist_ok=True)

# MLFlow model registry
config = configparser.ConfigParser()
config.read(Path(CONFIG_DIR, "mlflow.config"))
tracking_uri = config["secrets"]["MLFLOW_TRACKING_URI"]
mlflow.set_tracking_uri(tracking_uri)
tracking_username = config["secrets"]["MLFLOW_TRACKING_USERNAME"]
os.environ["MLFLOW_TRACKING_USERNAME"] = tracking_username
tracking_password = config["secrets"]["MLFLOW_TRACKING_PASSWORD"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = tracking_password

# Use config file to initialize rich logger
logging.config.fileConfig(Path(CONFIG_DIR, "logging.config"))
logger = logging.getLogger("root")
logger.handlers[0] = RichHandler(markup=True)

# Configure error formatter
pretty_errors.configure(
    separator_character="*",
    filename_display=pretty_errors.FILENAME_EXTENDED,
    line_number_first=True,
    display_link=True,
    lines_before=5,
    lines_after=2,
    line_color=pretty_errors.RED + "> " + pretty_errors.default_config.line_color,
    code_color="  " + pretty_errors.default_config.line_color,
    truncate_code=True,
    display_locals=True,
)
