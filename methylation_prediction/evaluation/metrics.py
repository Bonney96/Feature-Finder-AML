# evaluation/metrics.py
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

def evaluate_regression(predictions, labels, dataset_name):
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    logging.info(f'{dataset_name} MSE: {mse:.6f}')
    logging.info(f'{dataset_name} MAE: {mae:.6f}')
    logging.info(f'{dataset_name} R² Score: {r2:.6f}')
    print(f'{dataset_name} MSE: {mse:.6f}')
    print(f'{dataset_name} MAE: {mae:.6f}')
    print(f'{dataset_name} R² Score: {r2:.6f}')
    return mse, mae, r2
