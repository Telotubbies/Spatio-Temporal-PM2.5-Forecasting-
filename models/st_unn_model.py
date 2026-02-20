"""
Spatio-Temporal UNet (ST-UNN) model for PM2.5 forecasting.
"""
import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class STUNNPredictor(nn.Module):
    """
    ST-UNN model for grid-based PM2.5 forecasting.
    
    Architecture:
    - Encoder: ConvLSTM2D + Downsample
    - Bottleneck: ConvLSTM2D
    - Decoder: ConvTranspose3D + Skip connections
    - Output: Conv3D(1)
    """
    
    def __init__(
        self,
        input_channels: int,
        output_hours: int = 6,
        encoder_filters: int = 32,
        bottleneck_filters: int = 64,
        decoder_filters: int = 32,
        skip_connections: bool = True,
        grid_size: Tuple[int, int] = (32, 32)
    ):
        """
        Initialize ST-UNN model.
        
        Args:
            input_channels: Number of input channels
            output_hours: Number of hours to predict
            encoder_filters: Encoder filter size
            bottleneck_filters: Bottleneck filter size
            decoder_filters: Decoder filter size
            skip_connections: Use skip connections
            grid_size: Grid dimensions (H, W)
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.output_hours = output_hours
        self.skip_connections = skip_connections
        self.grid_size = grid_size
        
        # Encoder
        self.encoder_conv_lstm = ConvLSTM2D(
            in_channels=input_channels,
            out_channels=encoder_filters,
            kernel_size=3,
            padding=1
        )
        self.encoder_bn = nn.BatchNorm3d(encoder_filters)
        
        # Downsample
        self.downsample = nn.Conv3d(
            in_channels=encoder_filters,
            out_channels=encoder_filters,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2)
        )
        
        # Bottleneck
        self.bottleneck_conv_lstm = ConvLSTM2D(
            in_channels=encoder_filters,
            out_channels=bottleneck_filters,
            kernel_size=3,
            padding=1
        )
        self.bottleneck_bn = nn.BatchNorm3d(bottleneck_filters)
        
        # Decoder
        self.upsample = nn.ConvTranspose3d(
            in_channels=bottleneck_filters,
            out_channels=decoder_filters,
            kernel_size=(1, 2, 2),
            stride=(1, 2, 2)
        )
        
        if skip_connections:
            # Combine skip connection and upsampled features
            self.decoder_conv = nn.Conv3d(
                in_channels=decoder_filters + encoder_filters,
                out_channels=decoder_filters,
                kernel_size=3,
                padding=1
            )
        else:
            self.decoder_conv = nn.Conv3d(
                in_channels=decoder_filters,
                out_channels=decoder_filters,
                kernel_size=3,
                padding=1
            )
        
        self.decoder_bn = nn.BatchNorm3d(decoder_filters)
        
        # Output
        self.output_conv = nn.Conv3d(
            in_channels=decoder_filters,
            out_channels=1,
            kernel_size=3,
            padding=1
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, time, channels, H, W)
            
        Returns:
            Output tensor (batch, output_hours, 1, H, W)
        """
        # Encoder
        enc_out = self.encoder_conv_lstm(x)  # (batch, time, encoder_filters, H, W)
        enc_out = enc_out.permute(0, 2, 1, 3, 4)  # (batch, encoder_filters, time, H, W)
        enc_out = self.encoder_bn(enc_out)
        
        # Save for skip connection
        if self.skip_connections:
            skip = enc_out
        
        # Downsample
        enc_out = self.downsample(enc_out)  # (batch, encoder_filters, time, H/2, W/2)
        enc_out = enc_out.permute(0, 2, 1, 3, 4)  # (batch, time, encoder_filters, H/2, W/2)
        
        # Bottleneck
        bottleneck_out = self.bottleneck_conv_lstm(enc_out)  # (batch, time, bottleneck_filters, H/2, W/2)
        bottleneck_out = bottleneck_out.permute(0, 2, 1, 3, 4)  # (batch, bottleneck_filters, time, H/2, W/2)
        bottleneck_out = self.bottleneck_bn(bottleneck_out)
        
        # Upsample
        dec_out = self.upsample(bottleneck_out)  # (batch, decoder_filters, time, H, W)
        
        # Skip connection
        if self.skip_connections:
            # Crop skip if needed (in case of size mismatch)
            if skip.shape != dec_out.shape:
                # Center crop
                _, _, _, H, W = dec_out.shape
                _, _, _, H_skip, W_skip = skip.shape
                h_start = (H_skip - H) // 2
                w_start = (W_skip - W) // 2
                skip = skip[:, :, :, h_start:h_start+H, w_start:w_start+W]
            
            dec_out = torch.cat([dec_out, skip], dim=1)  # (batch, decoder_filters + encoder_filters, time, H, W)
        
        # Decoder
        dec_out = self.decoder_conv(dec_out)  # (batch, decoder_filters, time, H, W)
        dec_out = self.decoder_bn(dec_out)
        dec_out = torch.relu(dec_out)
        
        # Output
        output = self.output_conv(dec_out)  # (batch, 1, time, H, W)
        output = output.permute(0, 2, 1, 3, 4)  # (batch, time, 1, H, W)
        
        # Take only output_hours
        output = output[:, :self.output_hours, :, :, :]
        
        return output


# Import ConvLSTM2D from conv_lstm_model
from .conv_lstm_model import ConvLSTM2D

