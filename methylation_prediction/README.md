# Methylation Level Prediction Model

This project implements a machine learning model to predict DNA methylation levels in a given sequence. The model leverages histone mark features and sequence information to generate these predictions, with a modular design for training, evaluation, and interpretability.

---

## Project Overview

The goal of this project is to predict DNA methylation levels using sequence information and histone mark features. The model architecture is divided into two main modules:

- **DNAModule**: Processes DNA sequences using convolutional layers.
- **HistoneModule**: Processes histone mark features with a fully connected network.

These modules are then combined in the **JointModule** for a joint prediction of methylation levels. The project is structured for easy data loading, preprocessing, training, evaluation, and interpretability analysis.

---

## Directory Structure

The project is organized into a modular structure, each responsible for a different part of the pipeline:

```bash
methylation_prediction/
├── data/                          # Data loading and preprocessing
│   ├── data_loader.py             # Functions for loading data files
│   ├── preprocess.py              # Preprocessing and feature engineering functions
│   ├── dataset.py                 # PyTorch Dataset definition
├── models/                        # Model architecture components
│   ├── dna_module.py              # DNA sequence model
│   ├── histone_module.py          # Histone mark model
│   ├── joint_module.py            # Joint prediction model
├── training/                      # Training and hyperparameter optimization
│   ├── trainer.py                 # Training loop implementation
│   ├── hyperparameter_tuning.py   # Optuna-based hyperparameter tuning
├── evaluation/                    # Evaluation metrics and visualization
│   ├── metrics.py                 # Evaluation metrics (MSE, MAE, R²)
│   ├── visualization.py           # Plotting functions
├── interpretability/              # Model interpretability functions
│   ├── explainability.py          # Functions for interpretability analysis using Captum
├── utils/                         # Utility functions
│   ├── helpers.py                 # Helper functions for encoding, padding, etc.
│   ├── logging_config.py          # Logging configuration
├── config/                        # Configuration files
│   └── config.yaml                # Main configuration file
├── scripts/                       # Scripts to run training and evaluation
│   ├── run_training.py            # Script to run training and evaluation
├── notebooks/                     # Jupyter notebooks for data exploration and visualization
├── tests/                         # Unit tests for various components
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
```

python scripts/run_training.py
This script will:

Load and preprocess the data.
Run hyperparameter tuning using Optuna.
Train the model with the best hyperparameters.
Save the trained model and logging output to the results/ directory.
Evaluating the Model
The run_training.py script automatically evaluates the model on the test set and generates the following outputs:

Evaluation Metrics: MSE, MAE, and R² scores.
Visualizations: Scatter plots of predicted vs. actual methylation levels, training and validation loss curves, and interactive prediction plots saved in the results folder.
Running Interpretability Analysis
After training, the model’s interpretability is analyzed with saliency maps and feature attribution using Captum. The following interpretability methods are implemented:

Integrated Gradients: Attribution for individual sequences.
Gene-specific Saliency Maps: Saliency maps focused on sequences related to specific genes of interest.
To generate interpretability outputs, ensure run_explainability_methods is called within run_training.py. The generated saliency maps will be saved in the results directory.

Results
Upon successful completion, results are saved in a timestamped folder within the results/ directory. Key files include:

Model Checkpoints: Saved model states for reproducibility.
Plots and Visualizations: Prediction vs. actual plots, feature importance, saliency maps, and confusion matrices.
Training Logs: A comprehensive log of the training process, including hyperparameter settings and evaluation metrics.
Interpretability Reports: Saliency maps and gene-specific analyses to interpret model predictions.



License
This project is licensed under the MIT License. See LICENSE for more information.

Contact
For questions or collaboration inquiries, please contact the author via abonney@wustl.edu.
