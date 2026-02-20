"""
Visualization for PM2.5 forecasting evaluation.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger(__name__)


class Visualizer:
    """Visualize predictions and evaluation results."""
    
    def __init__(self, model_type: str, output_dir: Path):
        """
        Initialize visualizer.
        
        Args:
            model_type: Model type (lstm, conv_lstm, st_unn)
            output_dir: Output directory for plots
        """
        self.model_type = model_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timesteps: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None
    ):
        """
        Plot predictions vs ground truth.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            timesteps: Timestep indices (optional)
            save_path: Path to save plot
        """
        if self.model_type == "lstm":
            self._plot_station_predictions(y_true, y_pred, timesteps, save_path)
        else:
            self._plot_grid_predictions(y_true, y_pred, timesteps, save_path)
    
    def _plot_station_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timesteps: Optional[np.ndarray],
        save_path: Optional[Path]
    ):
        """Plot station-based predictions."""
        # Flatten
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Scatter plot
        axes[0, 0].scatter(y_true_flat, y_pred_flat, alpha=0.5)
        axes[0, 0].plot([y_true_flat.min(), y_true_flat.max()], 
                       [y_true_flat.min(), y_true_flat.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True PM2.5')
        axes[0, 0].set_ylabel('Predicted PM2.5')
        axes[0, 0].set_title('Prediction vs Ground Truth')
        axes[0, 0].grid(True)
        
        # Time series (sample)
        if timesteps is None:
            timesteps = np.arange(len(y_true_flat))
        
        sample_idx = np.random.choice(len(y_true_flat), min(1000, len(y_true_flat)), replace=False)
        axes[0, 1].plot(timesteps[sample_idx], y_true_flat[sample_idx], label='True', alpha=0.7)
        axes[0, 1].plot(timesteps[sample_idx], y_pred_flat[sample_idx], label='Predicted', alpha=0.7)
        axes[0, 1].set_xlabel('Sample')
        axes[0, 1].set_ylabel('PM2.5')
        axes[0, 1].set_title('Time Series (Sample)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Error distribution
        errors = y_pred_flat - y_true_flat
        axes[1, 0].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Error Distribution')
        axes[1, 0].axvline(0, color='r', linestyle='--', lw=2)
        axes[1, 0].grid(True)
        
        # Residual plot
        axes[1, 1].scatter(y_pred_flat, errors, alpha=0.5)
        axes[1, 1].axhline(0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Predicted PM2.5')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residual Plot')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'predictions.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def _plot_grid_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timesteps: Optional[np.ndarray],
        save_path: Optional[Path]
    ):
        """Plot grid-based predictions."""
        # Take first sample and first timestep
        if len(y_true.shape) == 5:
            y_true = y_true.squeeze(-1)
        if len(y_pred.shape) == 5:
            y_pred = y_pred.squeeze(-1)
        
        sample_idx = 0
        time_idx = 0
        
        true_grid = y_true[sample_idx, time_idx]
        pred_grid = y_pred[sample_idx, time_idx]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # True
        im1 = axes[0].imshow(true_grid, cmap='YlOrRd', origin='lower')
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        # Predicted
        im2 = axes[1].imshow(pred_grid, cmap='YlOrRd', origin='lower')
        axes[1].set_title('Predicted')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        # Error
        error_grid = pred_grid - true_grid
        im3 = axes[2].imshow(error_grid, cmap='RdBu_r', origin='lower', center=0)
        axes[2].set_title('Error (Predicted - True)')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'grid_predictions.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_training_curve(
        self,
        train_losses: list,
        val_losses: list,
        save_path: Optional[Path] = None
    ):
        """
        Plot training curve.
        
        Args:
            train_losses: Training losses
            val_losses: Validation losses
            save_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, label='Train Loss', marker='o')
        ax.plot(epochs, val_losses, label='Val Loss', marker='s')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Curve')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / 'training_curve.png', dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def create_report(
        self,
        metrics: dict,
        train_losses: list,
        val_losses: list,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Optional[Path] = None
    ):
        """
        Create comprehensive evaluation report (PDF).
        
        Args:
            metrics: Dictionary of metrics
            train_losses: Training losses
            val_losses: Validation losses
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save PDF
        """
        if save_path is None:
            save_path = self.output_dir / 'evaluation_report.pdf'
        
        with PdfPages(save_path) as pdf:
            # Page 1: Metrics summary
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axis('off')
            
            metrics_text = "Evaluation Metrics\n\n"
            for key, value in metrics.items():
                metrics_text += f"{key.upper()}: {value:.4f}\n"
            
            ax.text(0.5, 0.5, metrics_text, 
                   ha='center', va='center', fontsize=14, family='monospace')
            ax.set_title('Evaluation Metrics Summary', fontsize=16, fontweight='bold')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 2: Training curve
            self.plot_training_curve(train_losses, val_losses)
            # Re-open and add to PDF
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs = range(1, len(train_losses) + 1)
            ax.plot(epochs, train_losses, label='Train Loss', marker='o')
            ax.plot(epochs, val_losses, label='Val Loss', marker='s')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Training Curve')
            ax.legend()
            ax.grid(True)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Page 3: Predictions
            self.plot_predictions(y_true, y_pred)
            # Re-open and add to PDF
            if self.model_type == "lstm":
                self._plot_station_predictions(y_true, y_pred, None, None)
            else:
                self._plot_grid_predictions(y_true, y_pred, None, None)
            fig = plt.gcf()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Saved evaluation report to {save_path}")

