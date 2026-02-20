"""
Data loader for PM2.5 forecasting models.
"""
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple, Dict
from pathlib import Path

from preprocessing.grid_interpolation import GridInterpolator
from utils.sliding_window import create_sliding_window

logger = logging.getLogger(__name__)


class StationDataset(Dataset):
    """Dataset for station-based LSTM."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Input sequences (batch, time, features)
            y: Target sequences (batch, output_hours)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GridDataset(Dataset):
    """Dataset for grid-based models (ConvLSTM, ST-UNN)."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Input sequences (batch, time, H, W, channels)
            y: Target sequences (batch, output_hours, H, W, 1)
        """
        # Convert to (batch, time, channels, H, W) for PyTorch
        self.X = torch.FloatTensor(X).permute(0, 1, 4, 2, 3)  # (batch, time, channels, H, W)
        self.y = torch.FloatTensor(y).permute(0, 1, 4, 2, 3)  # (batch, output_hours, 1, H, W)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PM25DataLoader:
    """Data loader factory for different model types."""
    
    def __init__(
        self,
        model_type: str,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        """
        Initialize data loader.
        
        Args:
            model_type: Model type (lstm, conv_lstm, st_unn)
            batch_size: Batch size
            num_workers: Number of worker processes
            pin_memory: Pin memory for GPU
        """
        self.model_type = model_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
    def create_loaders(
        self,
        train_data: Tuple[np.ndarray, np.ndarray],
        val_data: Tuple[np.ndarray, np.ndarray],
        test_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, DataLoader]:
        """
        Create data loaders for train/val/test.
        
        Args:
            train_data: (X_train, y_train)
            val_data: (X_val, y_val)
            test_data: (X_test, y_test) optional
            
        Returns:
            Dictionary of data loaders
        """
        loaders = {}
        
        if self.model_type == "lstm":
            # Station-based
            train_dataset = StationDataset(train_data[0], train_data[1])
            val_dataset = StationDataset(val_data[0], val_data[1])
            
            loaders['train'] = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
            
            loaders['val'] = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
            
            if test_data:
                test_dataset = StationDataset(test_data[0], test_data[1])
                loaders['test'] = DataLoader(
                    test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory
                )
        
        else:
            # Grid-based (ConvLSTM, ST-UNN)
            train_dataset = GridDataset(train_data[0], train_data[1])
            val_dataset = GridDataset(val_data[0], val_data[1])
            
            loaders['train'] = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
            
            loaders['val'] = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory
            )
            
            if test_data:
                test_dataset = GridDataset(test_data[0], test_data[1])
                loaders['test'] = DataLoader(
                    test_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory
                )
        
        logger.info(f"Created data loaders for {self.model_type} model")
        return loaders

