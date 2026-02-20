"""
Monte Carlo Dropout for uncertainty estimation.
"""
import torch
import torch.nn as nn
from typing import Optional
import numpy as np


class MCDropout(nn.Module):
    """Monte Carlo Dropout wrapper."""
    
    def __init__(self, model: nn.Module, dropout_rate: float = 0.2):
        """
        Initialize MC Dropout.
        
        Args:
            model: Base model
            dropout_rate: Dropout rate
        """
        super().__init__()
        self.model = model
        self.dropout_rate = dropout_rate
        
        # Enable dropout during inference
        self._enable_dropout()
    
    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout3d):
                module.train()  # Keep dropout active
    
    def forward(self, x: torch.Tensor, n_samples: int = 100) -> tuple:
        """
        Forward pass with MC Dropout.
        
        Args:
            x: Input tensor
            n_samples: Number of Monte Carlo samples
            
        Returns:
            (mean_prediction, std_prediction)
        """
        self.model.eval()
        self._enable_dropout()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(x)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)  # (n_samples, batch, ...)
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        return mean_pred, std_pred

