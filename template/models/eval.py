from typing import Dict

from sklearn.metrics import (  # precision_recall_fscore_support,; matthews_corrcoef,; roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
)

from config.config import logger


def evaluation(labels, predictions) -> Dict:
    """
    Run the evaluation process
    """
    results = {}
    results["mse"] = mean_squared_error(labels, predictions)
    results["mae"] = mean_absolute_error(labels, predictions)
    # (
    #    results["precision"],
    #    results["recall"],
    #    results["fscore"],
    #    _,
    # ) = precision_recall_fscore_support(labels, predictions, average="weighted")
    # results["auc"] = roc_auc_score(labels, predictions, average="weighted")
    # results["mcc"] = matthews_corrcoef(labels, predictions)
    logger.info(f"Test results: {results}")
    return results
