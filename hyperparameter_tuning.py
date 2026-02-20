"""
Hyperparameter tuning using Optuna.
"""
import logging
import optuna
import torch
import torch.nn as nn
from typing import Dict
from pathlib import Path

from models import LSTMPredictor, ConvLSTMPredictor, STUNNPredictor
from training.trainer import ModelTrainer
from training.data_loader import PM25DataLoader

logger = logging.getLogger(__name__)


def objective(
    trial: optuna.Trial,
    model_type: str,
    train_loader,
    val_loader,
    device: torch.device,
    config: Dict
) -> float:
    """
    Optuna objective function.
    
    Args:
        trial: Optuna trial
        model_type: Model type
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device
        config: Configuration
        
    Returns:
        Validation loss
    """
    # Suggest hyperparameters
    if model_type == "lstm":
        hidden_size = trial.suggest_int("hidden_size", 32, 128, step=16)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        
        # Create model
        model = LSTMPredictor(
            input_features=config.get('input_features', 10),
            output_hours=config.get('output_hours', 6),
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    
    elif model_type == "conv_lstm":
        filters1 = trial.suggest_int("filters1", 16, 64, step=16)
        filters2 = trial.suggest_int("filters2", 32, 128, step=16)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        
        model = ConvLSTMPredictor(
            input_channels=config.get('input_channels', 10),
            output_hours=config.get('output_hours', 6),
            filters=[filters1, filters2],
            dropout=dropout
        )
    
    elif model_type == "st_unn":
        encoder_filters = trial.suggest_int("encoder_filters", 16, 64, step=16)
        bottleneck_filters = trial.suggest_int("bottleneck_filters", 32, 128, step=16)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        
        model = STUNNPredictor(
            input_channels=config.get('input_channels', 10),
            output_hours=config.get('output_hours', 6),
            encoder_filters=encoder_filters,
            bottleneck_filters=bottleneck_filters
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Update config
    training_config = config.copy()
    training_config['learning_rate'] = learning_rate
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        model_type=model_type,
        device=device,
        config=training_config
    )
    
    # Train for a few epochs
    epochs = 10  # Quick trial
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        early_stopping_patience=5
    )
    
    # Get validation loss
    val_loss = trainer.validate(val_loader)
    
    return val_loss


def tune_hyperparameters(
    model_type: str,
    train_loader,
    val_loader,
    device: torch.device,
    config: Dict,
    n_trials: int = 50,
    study_name: str = "pm25_forecasting"
):
    """
    Run hyperparameter tuning.
    
    Args:
        model_type: Model type
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device
        config: Configuration
        n_trials: Number of trials
        study_name: Study name
    """
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name
    )
    
    study.optimize(
        lambda trial: objective(trial, model_type, train_loader, val_loader, device, config),
        n_trials=n_trials
    )
    
    logger.info("Best hyperparameters:")
    logger.info(study.best_params)
    logger.info(f"Best validation loss: {study.best_value:.4f}")
    
    return study.best_params

