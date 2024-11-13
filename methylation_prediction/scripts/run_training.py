# scripts/run_training.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
import logging
from datetime import datetime
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

from data.data_loader import load_data_files
from data.preprocess import preprocess_data, standardize_chromosome_names
from data.dataset import MethylationDataset
from utils.helpers import set_seed, encode_sequences, get_device, init_weights
from utils.logging_config import setup_logging
from models.dna_module import DNAModule
from models.histone_module import HistoneModule
from models.joint_module import JointModule
from training.trainer import Trainer
from training.hyperparameter_tuning import run_hyperparameter_optimization
from evaluation.metrics import evaluate_regression
from evaluation.visualization import (
    plot_predictions_vs_actual,
    plot_training_validation_loss,
    plot_confusion_matrix,
    plot_histone_feature_importance,
    plot_saliency_maps,
    plot_gene_saliency_maps
)
from interpretability.explainability import (
    run_explainability_methods,
    compute_histone_feature_importance,
    compute_dmr_saliency_maps,
    compute_gene_saliency_maps
)
from utils.report_generator import generate_report
import copy

def main():
    # Set BASE_DIR to the root of the project
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Load configuration
    config_path = os.path.join(BASE_DIR, 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Adjust paths in config to be absolute
    config['data']['csv_file_path'] = os.path.join(BASE_DIR, config['data']['csv_file_path'])
    config['data']['bed_file_path'] = os.path.join(BASE_DIR, config['data']['bed_file_path'])
    config['data']['gene_annotations_bed_file_path'] = os.path.join(BASE_DIR, config['data']['gene_annotations_bed_file_path'])
    config['training']['model_save_path'] = os.path.join(BASE_DIR, config['training']['model_save_path'])
    config['logging']['log_file'] = os.path.join(BASE_DIR, config['logging']['log_file'])

    # Setup logging
    log_file = config['logging']['log_file']
    setup_logging(log_file)

    # Set random seed
    set_seed(42)

    # Device configuration
    device = get_device()

    # Load data files
    data, bed_df, gene_annotations_df = load_data_files(config)

    # Preprocess data
    data_with_genes = preprocess_data(data, bed_df, gene_annotations_df, config)

    # Filter data for specific sample
    unique_sample_value = config['data']['unique_sample_value']
    data_with_genes = data_with_genes[data_with_genes['Sample'] == unique_sample_value]
    logging.info(f"Data filtered to only include Sample {unique_sample_value}.")

    # Drop 'Sample' columns
    data_with_genes = data_with_genes.drop(columns=['Sample'])

    # Prepare features and labels
    histone_marks = config['data']['histone_marks']
    other_features = data_with_genes[histone_marks].values
    targets = data_with_genes['avg_methylation'].values

    # Split data
    X_other_temp, X_other_test, y_temp, y_test, data_temp, data_test = train_test_split(
        other_features, targets, data_with_genes, test_size=0.10, random_state=42)

    X_other_train, X_other_val, y_train, y_val, data_train, data_val = train_test_split(
        X_other_temp, y_temp, data_temp, test_size=0.1111, random_state=42)

    # Extract DMR identifiers for plotting
    train_dmrs = data_train['sequence_id'].values
    val_dmrs = data_val['sequence_id'].values
    test_dmrs = data_test['sequence_id'].values

    # Encode sequences
    max_seq_len = config['model']['max_seq_len']
    encoded_sequences_train = encode_sequences(data_train['sequence'].values, maxlen=max_seq_len)
    encoded_sequences_val = encode_sequences(data_val['sequence'].values, maxlen=max_seq_len)
    encoded_sequences_test = encode_sequences(data_test['sequence'].values, maxlen=max_seq_len)

    # Create Datasets
    train_dataset = MethylationDataset(encoded_sequences_train, X_other_train, y_train)
    val_dataset = MethylationDataset(encoded_sequences_val, X_other_val, y_val)
    test_dataset = MethylationDataset(encoded_sequences_test, X_other_test, y_test)

    # Histone input size
    histone_input_size = other_features.shape[1]  # Number of histone features
    config['model']['histone_input_size'] = histone_input_size

    # Run hyperparameter optimization
    best_params = run_hyperparameter_optimization(train_dataset, val_dataset, device, config)

    # Add num_epochs and model_save_path to best_params from config['training']
    best_params['num_epochs'] = config['training']['num_epochs']
    best_params['model_save_path'] = config['training']['model_save_path']

    # Deep copy of the full config to retain the nested structure
    model_config = copy.deepcopy(config)
    model_config['model'].update(best_params)  # Update only the 'model' section
    model_config['training']['batch_size'] = best_params['batch_size']
    model_config['training']['learning_rate'] = best_params['learning_rate']
    model_config['training']['num_epochs'] = best_params['num_epochs']
    model_config['training']['model_save_path'] = best_params['model_save_path']

    # Instantiate modules with best hyperparameters
    dna_module = DNAModule(model_config['model'])
    histone_module = HistoneModule(model_config['model'])
    model = JointModule(dna_module, histone_module, model_config['model'])
    model.apply(init_weights)
    model.to(device)

    # Define criterion, optimizer, scheduler
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['training']['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Initialize trainer and start training
    trainer = Trainer(model, criterion, optimizer, scheduler, device, model_config)
    train_loader = DataLoader(train_dataset, batch_size=model_config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=model_config['training']['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=model_config['training']['batch_size'])
    trainer.train(train_loader, val_loader)

    # Evaluate model
    train_predictions, train_labels = trainer.get_predictions(train_loader)
    val_predictions, val_labels = trainer.get_predictions(val_loader)
    test_predictions, test_labels = trainer.get_predictions(test_loader)

    evaluate_regression(train_predictions, train_labels, 'Training')
    evaluate_regression(val_predictions, val_labels, 'Validation')
    evaluate_regression(test_predictions, test_labels, 'Test')

    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_folder_path = os.path.join(BASE_DIR, 'results', f"run_{timestamp}")
    os.makedirs(new_folder_path, exist_ok=True)

    # Plotting
    plot_predictions_vs_actual(train_predictions, train_labels, train_dmrs, 'Training', new_folder_path)
    plot_predictions_vs_actual(val_predictions, val_labels, val_dmrs, 'Validation', new_folder_path)
    plot_predictions_vs_actual(test_predictions, test_labels, test_dmrs, 'Test', new_folder_path)
    plot_training_validation_loss(trainer.train_losses, trainer.val_losses, model_config['training']['num_epochs'], new_folder_path)

    # Run interpretability methods
    run_explainability_methods(model, test_loader, device, new_folder_path)

    # Compute feature importance
    feature_importance = compute_histone_feature_importance(model, test_loader, histone_marks, device)
    plot_histone_feature_importance(feature_importance, new_folder_path)

    # Compute DMR-level saliency maps
    sequences_list, dmr_saliency_maps = compute_dmr_saliency_maps(model, test_loader, device)
    plot_saliency_maps(sequences_list, dmr_saliency_maps, new_folder_path)

    # Compute gene saliency maps
    genes_of_interest = config['data']['genes_of_interest']
    compute_gene_saliency_maps(
        model, data_with_genes, genes_of_interest, histone_marks, model_config['model']['max_seq_len'], device, new_folder_path
    )

    # Plot confusion matrix
    plot_confusion_matrix(test_predictions, test_labels, new_folder_path)

    # Generate training report
    generate_report(
        model_config, trainer, train_predictions, train_labels, val_predictions, val_labels,
        test_predictions, test_labels, new_folder_path
    )

    # Save the model
    model_save_path = model_config['training']['model_save_path']
    torch.save(model.state_dict(), model_save_path)
    logging.info(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    main()
