"""
Station-based LSTM model for PM2.5 forecasting.
"""
import torch
import torch.nn as nn
from typing import Optional


class LSTMPredictor(nn.Module):
    """
    LSTM model for station-based PM2.5 forecasting.
    
    Architecture:
    - LSTM(64) with 2 layers
    - Dropout(0.2)
    - Dense(32)
    - Dense(output_hours)
    """
    
    def __init__(
        self,
        input_features: int,
        output_hours: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        dense_units: list = None
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_features: Number of input features
            output_hours: Number of hours to predict
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            dense_units: List of dense layer units
        """
        super().__init__()
        
        if dense_units is None:
            dense_units = [32]
        
        self.input_features = input_features
        self.output_hours = output_hours
        self.hidden_size = hidden_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Dense layers
        dense_layers = []
        prev_size = hidden_size
        
        for units in dense_units:
            dense_layers.append(nn.Linear(prev_size, units))
            dense_layers.append(nn.ReLU())
            dense_layers.append(nn.Dropout(dropout))
            prev_size = units
        
        # Output layer
        dense_layers.append(nn.Linear(prev_size, output_hours))
        
        self.dense = nn.Sequential(*dense_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, sequence_length, features)
            
        Returns:
            Output tensor (batch, output_hours)
        """
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Dropout
        last_hidden = self.dropout(last_hidden)
        
        # Dense layers
        output = self.dense(last_hidden)  # (batch, output_hours)
        
        return output

