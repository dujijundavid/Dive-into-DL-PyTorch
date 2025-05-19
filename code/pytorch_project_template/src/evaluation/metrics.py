"""
Evaluation metrics for PyTorch projects.
This module provides common evaluation metrics.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Metrics:
    """Collection of evaluation metrics."""
    
    @staticmethod
    def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate accuracy.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            
        Returns:
            Accuracy score
        """
        predictions = torch.argmax(outputs, dim=1)
        return accuracy_score(targets.cpu(), predictions.cpu())
        
    @staticmethod
    def precision(outputs: torch.Tensor, targets: torch.Tensor, average: str = 'macro') -> float:
        """
        Calculate precision.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            average: Averaging strategy
            
        Returns:
            Precision score
        """
        predictions = torch.argmax(outputs, dim=1)
        return precision_score(targets.cpu(), predictions.cpu(), average=average)
        
    @staticmethod
    def recall(outputs: torch.Tensor, targets: torch.Tensor, average: str = 'macro') -> float:
        """
        Calculate recall.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            average: Averaging strategy
            
        Returns:
            Recall score
        """
        predictions = torch.argmax(outputs, dim=1)
        return recall_score(targets.cpu(), predictions.cpu(), average=average)
        
    @staticmethod
    def f1(outputs: torch.Tensor, targets: torch.Tensor, average: str = 'macro') -> float:
        """
        Calculate F1 score.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            average: Averaging strategy
            
        Returns:
            F1 score
        """
        predictions = torch.argmax(outputs, dim=1)
        return f1_score(targets.cpu(), predictions.cpu(), average=average)
        
    @staticmethod
    def mse(outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate Mean Squared Error.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            
        Returns:
            MSE score
        """
        return torch.mean((outputs - targets) ** 2).item()
        
    @staticmethod
    def mae(outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Calculate Mean Absolute Error.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            
        Returns:
            MAE score
        """
        return torch.mean(torch.abs(outputs - targets)).item()
        
    @staticmethod
    def get_all_metrics(outputs: torch.Tensor, 
                       targets: torch.Tensor, 
                       task_type: str = 'classification') -> Dict[str, float]:
        """
        Calculate all relevant metrics for the given task.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            task_type: Type of task ('classification' or 'regression')
            
        Returns:
            Dictionary of metric names and values
        """
        metrics = {}
        
        if task_type == 'classification':
            metrics['accuracy'] = Metrics.accuracy(outputs, targets)
            metrics['precision'] = Metrics.precision(outputs, targets)
            metrics['recall'] = Metrics.recall(outputs, targets)
            metrics['f1'] = Metrics.f1(outputs, targets)
        elif task_type == 'regression':
            metrics['mse'] = Metrics.mse(outputs, targets)
            metrics['mae'] = Metrics.mae(outputs, targets)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
            
        return metrics 