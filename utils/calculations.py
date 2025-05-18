from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

def calculate_metrics(y_true, y_pred):
    """
    Calculate MAE, RMSE, and R² for model evaluation.
    Returns a dictionary with the metrics.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

def calculate_price_changes(predictions):
    """
    Calculate percentage price changes between consecutive days.
    Returns a pandas Series with percentage changes.
    """
    return pd.Series(predictions).pct_change() * 100