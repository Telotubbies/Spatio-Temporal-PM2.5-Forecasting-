"""
Training pipeline for PM2.5 forecasting models.
"""
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from datetime import datetime

from models import LSTMPredictor, ConvLSTMPredictor, STUNNPredictor

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trainer for PM2.5 forecasting models."""
    
    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        device: torch.device,
        config: dict
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            model_type: Model type (lstm, conv_lstm, st_unn)
            device: Device (cuda/cpu)
            config: Training configuration
        """
        self.model = model.to(device)
        self.model_type = model_type
        self.device = device
        self.config = config
        
        # Loss function
        if model_type == "st_unn":
            # Combined loss
            self.criterion = self._combined_loss
            self.mse_weight = config.get('loss_weights', {}).get('mse', 0.7)
            self.mae_weight = config.get('loss_weights', {}).get('mae', 0.3)
        else:
            self.criterion = nn.L1Loss()  # MAE
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.0001),
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        scheduler_type = config.get('lr_scheduler', 'ReduceLROnPlateau')
        if scheduler_type == 'ReduceLROnPlateau':
            scheduler_params = config.get('lr_scheduler_params', {})
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_params.get('factor', 0.5),
                patience=scheduler_params.get('patience', 5),
                min_lr=scheduler_params.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = None
        
        # Mixed precision
        self.use_amp = config.get('use_amp', True) and device.type == 'cuda'
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # TensorBoard
        tensorboard_dir = Path(config.get('tensorboard_dir', 'data/models/tensorboard'))
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(tensorboard_dir / datetime.now().strftime("%Y%m%d_%H%M%S")))
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', 'data/models/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_k = config.get('keep_last_k', 5)
        
        # Training state
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        self.global_step = 0
    
    def _combined_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Combined MSE + MAE loss for ST-UNN."""
        mse = nn.functional.mse_loss(pred, target)
        mae = nn.functional.l1_loss(pred, target)
        return self.mse_weight * mse + self.mae_weight * mae
    
    def train_epoch(self, train_loader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log to TensorBoard
            if batch_idx % self.config.get('log_interval', 10) == 0:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], self.global_step)
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def validate(self, val_loader) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        pred = self.model(x)
                        loss = self.criterion(pred, y)
                else:
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
    
    def train(
        self,
        train_loader,
        val_loader,
        epochs: int,
        early_stopping_patience: int = 10
    ):
        """
        Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Early stopping patience
        """
        logger.info(f"Starting training for {epochs} epochs")
        
        patience_counter = 0
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # Log
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            self.writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
            self.writer.add_scalar('Epoch/ValLoss', val_loss, epoch)
            
            # Checkpointing
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                patience_counter = 0
                self.save_checkpoint('best.pt', is_best=True)
            else:
                patience_counter += 1
            
            # Save last
            if self.config.get('save_last', True):
                self.save_checkpoint('last.pt', is_best=False)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info("Training completed")
        self.writer.close()
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            logger.info(f"Saved best model to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.current_epoch = checkpoint.get('epoch', 0)
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")

