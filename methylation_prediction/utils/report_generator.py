# utils/report_generator.py

import os
from datetime import datetime

def generate_report(config, trainer, train_predictions, train_labels, val_predictions, val_labels, test_predictions, test_labels, folder_path):
    train_mse = trainer.train_losses[-1]
    val_mse = trainer.val_losses[-1]
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    test_mse = mean_squared_error(test_labels, test_predictions)
    test_mae = mean_absolute_error(test_labels, test_predictions)
    test_r2 = r2_score(test_labels, test_predictions)

    current_time = datetime.now()

    report_content = f"""
Methylation Level Prediction Report
===================================

Date and Time: {current_time.strftime('%d %B %Y, %H:%M')}

Hyperparameters Used:
---------------------
{config['model']}

Training Results:
-----------------
- Training MSE: {train_mse:.6f}

Validation Results:
-------------------
- Validation MSE: {val_mse:.6f}

Test Results:
-------------
- Test MSE: {test_mse:.6f}
- Test MAE: {test_mae:.6f}
- Test R² Score: {test_r2:.6f}
"""

    # Save the report
    report_filename = os.path.join(folder_path, 'training_report.txt')
    with open(report_filename, 'w') as f:
        f.write(report_content)
    print(f"Training report saved at: {report_filename}")
