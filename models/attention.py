"""
Attention mechanisms for spatio-temporal models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SpatialAttention(nn.Module):
    """Spatial attention module."""
    
    def __init__(self, in_channels: int):
        """
        Initialize spatial attention.
        
        Args:
            in_channels: Input channels
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention.
        
        Args:
            x: Input tensor (batch, channels, H, W)
            
        Returns:
            Attended tensor
        """
        # Compute attention map
        attention = self.conv(x)  # (batch, 1, H, W)
        attention = torch.sigmoid(attention)
        
        # Apply attention
        attended = x * attention
        
        return attended


class TemporalAttention(nn.Module):
    """Temporal attention module."""
    
    def __init__(self, in_channels: int):
        """
        Initialize temporal attention.
        
        Args:
            in_channels: Input channels
        """
        super().__init__()
        self.conv = nn.Conv1d(in_channels, 1, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal attention.
        
        Args:
            x: Input tensor (batch, time, channels, H, W)
            
        Returns:
            Attended tensor
        """
        batch, time, channels, H, W = x.shape
        
        # Reshape to (batch, channels, time*H*W)
        x_flat = x.permute(0, 2, 1, 3, 4).contiguous()
        x_flat = x_flat.view(batch, channels, time * H * W)
        
        # Compute attention
        attention = self.conv(x_flat)  # (batch, 1, time*H*W)
        attention = attention.view(batch, 1, time, H, W)
        attention = torch.sigmoid(attention)
        
        # Apply attention
        attended = x * attention
        
        return attended


class SpatioTemporalAttention(nn.Module):
    """Combined spatio-temporal attention."""
    
    def __init__(self, in_channels: int):
        """
        Initialize spatio-temporal attention.
        
        Args:
            in_channels: Input channels
        """
        super().__init__()
        self.spatial_attn = SpatialAttention(in_channels)
        self.temporal_attn = TemporalAttention(in_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatio-temporal attention.
        
        Args:
            x: Input tensor (batch, time, channels, H, W)
            
        Returns:
            Attended tensor
        """
        # Temporal attention
        x = self.temporal_attn(x)
        
        # Spatial attention (apply to each timestep)
        batch, time, channels, H, W = x.shape
        x = x.view(batch * time, channels, H, W)
        x = self.spatial_attn(x)
        x = x.view(batch, time, channels, H, W)
        
        return x

