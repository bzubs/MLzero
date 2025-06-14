import numpy as np

def mean_absolute_error(y_true, y_pred):
    """Mean Absolute Error (MAE)"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    """Mean Squared Error (MSE)"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    """R^2 (coefficient of determination) regression score function"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
