"""
Main training script for PM2.5 forecasting models.
"""
import logging
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import argparse
from datetime import datetime

from config import PipelineConfig
from preprocessing.normalizer import DataNormalizer
from preprocessing.grid_interpolation import GridInterpolator
from utils.tensor_builder import TensorBuilder
from training.data_loader import PM25DataLoader
from training.trainer import ModelTrainer
from models import LSTMPredictor, ConvLSTMPredictor, STUNNPredictor
from evaluation.metrics import Evaluator
from evaluation.visualization import Visualizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load processed data."""
    processed_dir = Path(data_dir) / "processed"
    
    # Load all parquet files
    parquet_files = list(processed_dir.glob("**/*.parquet"))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {processed_dir}")
    
    logger.info(f"Loading {len(parquet_files)} parquet files...")
    
    dfs = []
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to load {file}: {e}")
    
    if not dfs:
        raise ValueError("No data loaded")
    
    df = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(df)} records")
    
    return df


def prepare_data(
    df: pd.DataFrame,
    model_type: str,
    config: Dict
) -> tuple:
    """
    Prepare data for training.
    
    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    # Feature columns
    exclude_cols = ["timestamp", "station_id", "lat", "lon", "year", "month"]
    feature_cols = [col for col in df.columns if col not in exclude_cols and col != "pm25"]
    target_col = "pm25"
    
    logger.info(f"Feature columns: {feature_cols}")
    
    # Normalize
    normalizer = DataNormalizer(
        scaler_type=config.get('features', {}).get('scaler', 'StandardScaler')
    )
    df_normalized = normalizer.fit_transform(df, feature_cols)
    
    # Save scaler
    scaler_path = Path(config.get('training', {}).get('checkpoint_dir', 'data/models/checkpoints')) / 'scaler.pkl'
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    normalizer.save(scaler_path)
    
    # Time-based split
    df_normalized = df_normalized.sort_values('timestamp')
    split_method = config.get('training', {}).get('split_method', 'time_based')
    
    if split_method == 'time_based':
        train_split = config.get('training', {}).get('train_split', 0.7)
        val_split = config.get('training', {}).get('val_split', 0.15)
        
        n = len(df_normalized)
        train_end = int(n * train_split)
        val_end = int(n * (train_split + val_split))
        
        df_train = df_normalized.iloc[:train_end]
        df_val = df_normalized.iloc[train_end:val_end]
        df_test = df_normalized.iloc[val_end:]
    else:
        # Random split
        df_normalized = df_normalized.sample(frac=1, random_state=42)
        train_split = config.get('training', {}).get('train_split', 0.7)
        val_split = config.get('training', {}).get('val_split', 0.15)
        
        n = len(df_normalized)
        train_end = int(n * train_split)
        val_end = int(n * (train_split + val_split))
        
        df_train = df_normalized.iloc[:train_end]
        df_val = df_normalized.iloc[train_end:val_end]
        df_test = df_normalized.iloc[val_end:]
    
    logger.info(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")
    
    # Build tensors
    input_hours = config.get('model', {}).get('input_hours', 24)
    output_hours = config.get('model', {}).get('output_hours', 6)
    grid_size = tuple(config.get('grid', {}).get('size', 32) for _ in range(2))
    bbox = tuple(config.get('data', {}).get('bbox', {}).values())
    
    tensor_builder = TensorBuilder(
        model_type=model_type,
        input_hours=input_hours,
        output_hours=output_hours,
        grid_size=grid_size,
        bbox=bbox if model_type != 'lstm' else None,
        interpolation_method=config.get('grid', {}).get('interpolation_method', 'idw')
    )
    
    X_train, y_train = tensor_builder.build_tensors(df_train, feature_cols, target_col)
    X_val, y_val = tensor_builder.build_tensors(df_val, feature_cols, target_col)
    X_test, y_test = tensor_builder.build_tensors(df_test, feature_cols, target_col)
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), normalizer


def create_model(model_type: str, config: Dict, input_features: int = None) -> torch.nn.Module:
    """Create model based on type."""
    model_config = config.get('model', {})
    input_hours = model_config.get('input_hours', 24)
    output_hours = model_config.get('output_hours', 6)
    grid_size = tuple(config.get('grid', {}).get('size', 32) for _ in range(2))
    
    if model_type == "lstm":
        if input_features is None:
            raise ValueError("input_features required for LSTM")
        
        lstm_config = model_config.get('lstm', {})
        model = LSTMPredictor(
            input_features=input_features,
            output_hours=output_hours,
            hidden_size=lstm_config.get('hidden_size', 64),
            num_layers=lstm_config.get('num_layers', 2),
            dropout=lstm_config.get('dropout', 0.2),
            dense_units=lstm_config.get('dense_units', [32])
        )
    
    elif model_type == "conv_lstm":
        if input_features is None:
            raise ValueError("input_features (channels) required for ConvLSTM")
        
        conv_lstm_config = model_config.get('conv_lstm', {})
        model = ConvLSTMPredictor(
            input_channels=input_features,
            output_hours=output_hours,
            filters=conv_lstm_config.get('filters', [32, 64]),
            kernel_size=conv_lstm_config.get('kernel_size', 3),
            dropout=conv_lstm_config.get('dropout', 0.2),
            grid_size=grid_size
        )
    
    elif model_type == "st_unn":
        if input_features is None:
            raise ValueError("input_features (channels) required for ST-UNN")
        
        st_unn_config = model_config.get('st_unn', {})
        model = STUNNPredictor(
            input_channels=input_features,
            output_hours=output_hours,
            encoder_filters=st_unn_config.get('encoder_filters', 32),
            bottleneck_filters=st_unn_config.get('bottleneck_filters', 64),
            decoder_filters=st_unn_config.get('decoder_filters', 32),
            skip_connections=st_unn_config.get('skip_connections', True),
            grid_size=grid_size
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Train PM2.5 forecasting model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--model-type', type=str, choices=['lstm', 'conv_lstm', 'st_unn'], 
                       default='st_unn', help='Model type')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(Path(args.config))
    model_type = args.model_type or config.get('model', {}).get('type', 'st_unn')
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Model type: {model_type}")
    
    # Load data
    df = load_data(args.data_dir)
    
    # Prepare data
    train_data, val_data, test_data, normalizer = prepare_data(df, model_type, config)
    
    # Get input features
    if model_type == "lstm":
        input_features = train_data[0].shape[2]  # (batch, time, features)
    else:
        input_features = train_data[0].shape[4]  # (batch, time, H, W, channels)
    
    # Create model
    model = create_model(model_type, config, input_features)
    
    # Update config with model info
    training_config = config.get('training', {})
    training_config['input_features'] = input_features
    training_config['input_channels'] = input_features if model_type != 'lstm' else None
    training_config['grid_size'] = tuple(config.get('grid', {}).get('size', 32) for _ in range(2))
    training_config['loss_weights'] = config.get('model', {}).get('st_unn', {}).get('loss_weights', {})
    
    # Create data loaders
    data_loader = PM25DataLoader(
        model_type=model_type,
        batch_size=training_config.get('batch_size', 8),
        num_workers=config.get('performance', {}).get('num_workers', 4),
        pin_memory=config.get('performance', {}).get('pin_memory', True)
    )
    
    loaders = data_loader.create_loaders(train_data, val_data, test_data)
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        model_type=model_type,
        device=device,
        config=training_config
    )
    
    # Train
    trainer.train(
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        epochs=training_config.get('epochs', 100),
        early_stopping_patience=training_config.get('early_stopping_patience', 10)
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loss = trainer.validate(loaders['test'])
    logger.info(f"Test Loss: {test_loss:.4f}")
    
    # Get predictions for evaluation
    trainer.model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in loaders['test']:
            x = x.to(device)
            y = y.to(device)
            pred = trainer.model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    
    # Compute metrics
    evaluator = Evaluator(model_type=model_type)
    metrics = evaluator.compute_metrics(y_true, y_pred)
    
    logger.info("Evaluation Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Visualize
    output_dir = Path(config.get('evaluation', {}).get('plot_dir', 'data/models/evaluation'))
    visualizer = Visualizer(model_type=model_type, output_dir=output_dir)
    
    # Create report
    visualizer.create_report(
        metrics=metrics,
        train_losses=[],  # Would need to track during training
        val_losses=[],
        y_true=y_true,
        y_pred=y_pred
    )
    
    logger.info("Training and evaluation completed!")


if __name__ == "__main__":
    main()

