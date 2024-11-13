# training/hyperparameter_tuning.py

import optuna
import torch
from torch.utils.data import DataLoader
from models.dna_module import DNAModule
from models.histone_module import HistoneModule
from models.joint_module import JointModule
from utils.helpers import init_weights
from training.trainer import Trainer
import logging

def objective(trial, train_dataset, val_dataset, device, config):
    # Copy the model config to avoid modifying it
    model_config = config['model'].copy()
    training_config = config['training'].copy()

    # Suggest hyperparameters
    model_config['embedding_dim'] = trial.suggest_int('embedding_dim', 16, 64, step=16)
    model_config['num_filters_conv1'] = trial.suggest_int('num_filters_conv1', 32, 128, step=32)
    model_config['num_filters_conv2'] = trial.suggest_int('num_filters_conv2', 64, 256, step=64)
    model_config['kernel_size_conv1'] = trial.suggest_int('kernel_size_conv1', 3, 7, step=2)
    model_config['kernel_size_conv2'] = trial.suggest_int('kernel_size_conv2', 3, 7, step=2)
    model_config['fc_size_dna'] = trial.suggest_int('fc_size_dna', 128, 512, step=128)
    model_config['histone_hidden_size'] = trial.suggest_int('histone_hidden_size', 64, 256, step=64)
    model_config['joint_hidden_size'] = trial.suggest_int('joint_hidden_size', 128, 512, step=128)
    model_config['dropout_rate'] = trial.suggest_float('dropout_rate', 0.1, 0.7)
    model_config['activation_func'] = trial.suggest_categorical('activation_func', ['relu', 'leaky_relu', 'tanh'])
    training_config['learning_rate'] = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    training_config['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'])

    # Instantiate modules
    dna_module = DNAModule(model_config)
    histone_module = HistoneModule(model_config)
    model = JointModule(dna_module, histone_module, model_config)

    # Initialize weights
    model.apply(init_weights)
    model.to(device)

    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Initialize trainer
    trial_config = {'training': {'num_epochs': 10}}  # Use 10 epochs for hyperparameter optimization
    trainer = Trainer(model, criterion, optimizer, scheduler, device, trial_config)

    # Train the model
    trainer.train(train_loader, val_loader)

    # Validate the model
    val_loss = trainer.validate(val_loader)

    return val_loss

def run_hyperparameter_optimization(train_dataset, val_dataset, device, config):
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, train_dataset, val_dataset, device, config),
        n_trials=config['training']['optuna_trials']
    )
    logging.info(f"Best trial: {study.best_trial.number}")
    logging.info(f"Best value (validation loss): {study.best_trial.value}")
    logging.info(f"Best hyperparameters: {study.best_trial.params}")
    return study.best_trial.params
