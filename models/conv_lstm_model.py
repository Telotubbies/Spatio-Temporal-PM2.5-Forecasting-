"""
Grid-based ConvLSTM model for PM2.5 forecasting.
"""
import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class ConvLSTMPredictor(nn.Module):
    """
    ConvLSTM model for grid-based PM2.5 forecasting.
    
    Architecture:
    - ConvLSTM2D(32 filters)
    - BatchNorm
    - ConvLSTM2D(64 filters)
    - Conv3D(1 filter)
    """
    
    def __init__(
        self,
        input_channels: int,
        output_hours: int = 6,
        filters: List[int] = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        grid_size: Tuple[int, int] = (32, 32)
    ):
        """
        Initialize ConvLSTM model.
        
        Args:
            input_channels: Number of input channels
            output_hours: Number of hours to predict
            filters: List of filter sizes
            kernel_size: Convolution kernel size
            dropout: Dropout rate
            grid_size: Grid dimensions (H, W)
        """
        super().__init__()
        
        if filters is None:
            filters = [32, 64]
        
        self.input_channels = input_channels
        self.output_hours = output_hours
        self.grid_size = grid_size
        
        # ConvLSTM layers
        self.conv_lstm1 = ConvLSTM2D(
            in_channels=input_channels,
            out_channels=filters[0],
            kernel_size=kernel_size,
            padding=1
        )
        
        self.bn1 = nn.BatchNorm3d(filters[0])
        
        self.conv_lstm2 = ConvLSTM2D(
            in_channels=filters[0],
            out_channels=filters[1],
            kernel_size=kernel_size,
            padding=1
        )
        
        self.bn2 = nn.BatchNorm3d(filters[1])
        
        self.dropout = nn.Dropout3d(dropout)
        
        # Output layer
        self.output_conv = nn.Conv3d(
            in_channels=filters[1],
            out_channels=1,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, time, channels, H, W)
            
        Returns:
            Output tensor (batch, output_hours, 1, H, W)
        """
        # ConvLSTM 1
        x = self.conv_lstm1(x)  # (batch, time, filters[0], H, W)
        x = x.permute(0, 2, 1, 3, 4)  # (batch, filters[0], time, H, W)
        x = self.bn1(x)
        x = x.permute(0, 2, 1, 3, 4)  # (batch, time, filters[0], H, W)
        
        # ConvLSTM 2
        x = self.conv_lstm2(x)  # (batch, time, filters[1], H, W)
        x = x.permute(0, 2, 1, 3, 4)  # (batch, filters[1], time, H, W)
        x = self.bn2(x)
        x = self.dropout(x)
        
        # Output
        x = self.output_conv(x)  # (batch, 1, output_hours, H, W)
        x = x.permute(0, 2, 1, 3, 4)  # (batch, output_hours, 1, H, W)
        
        return x


class ConvLSTM2D(nn.Module):
    """
    ConvLSTM2D cell implementation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1
    ):
        """
        Initialize ConvLSTM2D cell.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            padding: Padding
        """
        super().__init__()
        
        self.out_channels = out_channels
        
        # Convolutions for input, forget, cell, output gates
        self.conv_i = nn.Conv2d(
            in_channels + out_channels,
            out_channels,
            kernel_size,
            padding=padding
        )
        self.conv_f = nn.Conv2d(
            in_channels + out_channels,
            out_channels,
            kernel_size,
            padding=padding
        )
        self.conv_c = nn.Conv2d(
            in_channels + out_channels,
            out_channels,
            kernel_size,
            padding=padding
        )
        self.conv_o = nn.Conv2d(
            in_channels + out_channels,
            out_channels,
            kernel_size,
            padding=padding
        )
    
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, time, channels, H, W) or (batch, channels, H, W)
            hidden: Hidden state (h, c)
            
        Returns:
            Output tensor (batch, time, channels, H, W)
        """
        if len(x.shape) == 4:
            # Single timestep
            x = x.unsqueeze(1)  # (batch, 1, channels, H, W)
        
        batch_size, time_steps, channels, H, W = x.shape
        
        if hidden is None:
            h = torch.zeros(batch_size, self.out_channels, H, W, device=x.device)
            c = torch.zeros(batch_size, self.out_channels, H, W, device=x.device)
        else:
            h, c = hidden
        
        outputs = []
        
        for t in range(time_steps):
            # Current input
            x_t = x[:, t, :, :, :]  # (batch, channels, H, W)
            
            # Concatenate input and hidden
            combined = torch.cat([x_t, h], dim=1)  # (batch, in_channels + out_channels, H, W)
            
            # Gates
            i = torch.sigmoid(self.conv_i(combined))
            f = torch.sigmoid(self.conv_f(combined))
            c = f * c + i * torch.tanh(self.conv_c(combined))
            o = torch.sigmoid(self.conv_o(combined))
            h = o * torch.tanh(c)
            
            outputs.append(h)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # (batch, time, out_channels, H, W)
        
        return output

