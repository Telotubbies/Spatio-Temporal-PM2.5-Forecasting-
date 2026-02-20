"""
Evaluation metrics for PM2.5 forecasting.
"""
import logging
import numpy as np
import torch
from typing import Dict, Optional
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate model predictions."""
    
    def __init__(self, model_type: str):
        """
        Initialize evaluator.
        
        Args:
            model_type: Model type (lstm, conv_lstm, st_unn)
        """
        self.model_type = model_type
    
    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Flatten for station-based models
        if self.model_type == "lstm":
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
        else:
            # Grid-based: flatten spatial dimensions
            y_true_flat = y_true.flatten()
            y_pred_flat = y_pred.flatten()
        
        # Basic metrics
        metrics['mae'] = self.mae(y_true_flat, y_pred_flat)
        metrics['rmse'] = self.rmse(y_true_flat, y_pred_flat)
        metrics['r2'] = self.r2(y_true_flat, y_pred_flat)
        metrics['mape'] = self.mape(y_true_flat, y_pred_flat)
        
        # Grid-specific metrics
        if self.model_type in ["conv_lstm", "st_unn"]:
            metrics['pixel_mae'] = self.pixel_mae(y_true, y_pred)
            metrics['ssim'] = self.ssim(y_true, y_pred)
        
        return metrics
    
    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))
    
    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R² score."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)
    
    def mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        mask = y_true != 0
        if mask.sum() == 0:
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def pixel_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Pixel-wise MAE for grid predictions.
        
        Args:
            y_true: (batch, time, H, W, 1) or (batch, time, H, W)
            y_pred: (batch, time, H, W, 1) or (batch, time, H, W)
        """
        if len(y_true.shape) == 5:
            y_true = y_true.squeeze(-1)
        if len(y_pred.shape) == 5:
            y_pred = y_pred.squeeze(-1)
        
        # Average over spatial dimensions
        pixel_errors = np.mean(np.abs(y_true - y_pred), axis=(2, 3))  # (batch, time)
        return np.mean(pixel_errors)
    
    def ssim(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Structural Similarity Index for grid predictions.
        
        Args:
            y_true: (batch, time, H, W, 1) or (batch, time, H, W)
            y_pred: (batch, time, H, W, 1) or (batch, time, H, W)
        """
        if len(y_true.shape) == 5:
            y_true = y_true.squeeze(-1)
        if len(y_pred.shape) == 5:
            y_pred = y_pred.squeeze(-1)
        
        ssim_values = []
        
        # Compute SSIM for each sample and timestep
        batch_size, time_steps = y_true.shape[:2]
        
        for b in range(batch_size):
            for t in range(time_steps):
                true_img = y_true[b, t]
                pred_img = y_pred[b, t]
                
                # Normalize to [0, 1] for SSIM
                true_min, true_max = true_img.min(), true_img.max()
                pred_min, pred_max = pred_img.min(), pred_img.max()
                
                if true_max - true_min > 0:
                    true_norm = (true_img - true_min) / (true_max - true_min)
                else:
                    true_norm = true_img
                
                if pred_max - pred_min > 0:
                    pred_norm = (pred_img - pred_min) / (pred_max - pred_min)
                else:
                    pred_norm = pred_img
                
                # Compute SSIM
                ssim_val = ssim(
                    true_norm,
                    pred_norm,
                    data_range=1.0
                )
                ssim_values.append(ssim_val)
        
        return np.mean(ssim_values)

