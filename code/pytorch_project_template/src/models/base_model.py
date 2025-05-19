"""
Base model module for PyTorch projects.
This module provides a base class for all models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple

class BaseModel(nn.Module):
    """Base model class that implements common functionality."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model.
        
        Args:
            config: Model configuration dictionary
        """
        super().__init__()
        self.config = config
        self._build_model()
        
    def _build_model(self):
        """Build the model architecture."""
        raise NotImplementedError
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        raise NotImplementedError
        
    def get_loss(self, 
                 outputs: torch.Tensor, 
                 targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            
        Returns:
            Loss tensor
        """
        raise NotImplementedError
        
    def get_metrics(self, 
                   outputs: torch.Tensor, 
                   targets: torch.Tensor) -> Dict[str, float]:
        """
        Calculate evaluation metrics.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            
        Returns:
            Dictionary of metric names and values
        """
        raise NotImplementedError
        
    def save_checkpoint(self, path: str):
        """
        Save model checkpoint.
        
        Args:
            path: Path to save the checkpoint
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, path)
        
    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.
        
        Args:
            path: Path to the checkpoint
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config = checkpoint['config']
        
class ModelFactory:
    """Factory class for creating model instances."""
    
    @staticmethod
    def create_model(model_name: str, config: Dict[str, Any]) -> BaseModel:
        """
        Create a model instance.
        
        Args:
            model_name: Name of the model to create
            config: Model configuration
            
        Returns:
            Model instance
        """
        # This should be implemented based on your specific models
        raise NotImplementedError 