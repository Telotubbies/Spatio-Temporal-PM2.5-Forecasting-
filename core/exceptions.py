"""
Custom exception hierarchy for PM2.5 forecasting pipeline.
"""
from typing import Optional


class PipelineError(Exception):
    """Base exception for all pipeline errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        """
        Initialize pipeline error.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class DataCollectionError(PipelineError):
    """Error during data collection from external APIs."""
    pass


class DataValidationError(PipelineError):
    """Error during data validation."""
    pass


class FeatureEngineeringError(PipelineError):
    """Error during feature engineering."""
    pass


class StorageError(PipelineError):
    """Error during data storage operations."""
    pass


class ConfigurationError(PipelineError):
    """Error in configuration."""
    pass

