# Copyright (c) Hebes Intelligence Private Company

import numpy as np
from sklearn.metrics import root_mean_squared_error
from sklearn.utils.validation import column_or_1d


def cvrmse(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None):
    """Compute the Coefficient of Variation of the Root Mean Squared Error.

    Args:
        y_true (numpy.ndarray): Ground truth (correct) target values.
        y_pred (numpy.ndarray): Estimated target values.
        sample_weight(numpy.ndarray, optional): The weight of each observation in the
            input data. Defaults to None.

    Returns:
        float: The metric's value.
    """

    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)

    return float(
        root_mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
        / np.mean(y_true)
    )


def nmbe(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None):
    """Compute the Normalized Mean Bias Error.

    Args:
        y_true (numpy.ndarray): Ground truth (correct) target values.
        y_pred (numpy.ndarray): Estimated target values.
        sample_weight(numpy.ndarray, optional): The weight of each observation in the
            input data. Defaults to None.

    Returns:
        float: The metric's value.
    """

    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)

    return float(
        np.average(y_true - y_pred, weights=sample_weight, axis=0) / np.mean(y_true)
    )

