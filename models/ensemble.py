"""
Ensemble models for improved predictions.
"""
import torch
import torch.nn as nn
from typing import List, Dict
import numpy as np


class EnsemblePredictor(nn.Module):
    """Ensemble of multiple models."""
    
    def __init__(
        self,
        models: List[nn.Module],
        weights: Optional[List[float]] = None,
        method: str = "weighted_average"
    ):
        """
        Initialize ensemble.
        
        Args:
            models: List of trained models
            weights: Weights for each model (if None, equal weights)
            method: Ensemble method (weighted_average, voting)
        """
        super().__init__()
        self.models = nn.ModuleList(models)
        self.method = method
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        self.register_buffer('weights', torch.tensor(weights))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Ensemble prediction
        """
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Stack predictions
        pred_stack = torch.stack(predictions, dim=0)  # (n_models, batch, ...)
        
        if self.method == "weighted_average":
            # Weighted average
            weights = self.weights.view(-1, *([1] * (len(pred_stack.shape) - 1)))
            ensemble_pred = torch.sum(pred_stack * weights, dim=0)
        
        elif self.method == "voting":
            # Majority voting (for classification)
            ensemble_pred = torch.mode(pred_stack, dim=0)[0]
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")
        
        return ensemble_pred
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100
    ) -> tuple:
        """
        Predict with uncertainty estimation using ensemble.
        
        Args:
            x: Input tensor
            n_samples: Number of samples for uncertainty
            
        Returns:
            (mean_prediction, std_prediction)
        """
        all_predictions = []
        
        for _ in range(n_samples):
            # Get predictions from all models
            predictions = []
            for model in self.models:
                model.eval()
                with torch.no_grad():
                    pred = model(x)
                    predictions.append(pred.cpu().numpy())
            
            all_predictions.append(np.mean(predictions, axis=0))
        
        all_predictions = np.array(all_predictions)  # (n_samples, batch, ...)
        
        mean_pred = np.mean(all_predictions, axis=0)
        std_pred = np.std(all_predictions, axis=0)
        
        return mean_pred, std_pred

