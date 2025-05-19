"""
Training utilities for PyTorch projects.
This module provides a standardized training pipeline.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
import logging
from tqdm import tqdm
import os

class Trainer:
    """Training pipeline for PyTorch models."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            config: Training configuration
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        metrics = {}
        
        with tqdm(self.train_loader, desc='Training') as pbar:
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.model.get_loss(output, target)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
                
        metrics['train_loss'] = total_loss / len(self.train_loader)
        return metrics
        
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary of validation metrics
        """
        if not self.val_loader:
            return {}
            
        self.model.eval()
        total_loss = 0
        metrics = {}
        
        with torch.no_grad():
            with tqdm(self.val_loader, desc='Validation') as pbar:
                for data, target in pbar:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = self.model.get_loss(output, target)
                    total_loss += loss.item()
                    
                    # Calculate metrics
                    batch_metrics = self.model.get_metrics(output, target)
                    for k, v in batch_metrics.items():
                        metrics[k] = metrics.get(k, 0) + v
                        
        # Average metrics
        metrics = {k: v / len(self.val_loader) for k, v in metrics.items()}
        metrics['val_loss'] = total_loss / len(self.val_loader)
        return metrics
        
    def train(self, num_epochs: int, save_dir: Optional[str] = None):
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save checkpoints
        """
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.logger.info(f'Epoch {epoch+1}/{num_epochs}')
            
            # Train
            train_metrics = self.train_epoch()
            self.logger.info(f'Training metrics: {train_metrics}')
            
            # Validate
            val_metrics = self.validate()
            if val_metrics:
                self.logger.info(f'Validation metrics: {val_metrics}')
                
                # Save best model
                if save_dir and val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    os.makedirs(save_dir, exist_ok=True)
                    self.model.save_checkpoint(
                        os.path.join(save_dir, 'best_model.pth')
                    )
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
                
    def predict(self, data: torch.Tensor) -> torch.Tensor:
        """
        Make predictions.
        
        Args:
            data: Input data
            
        Returns:
            Model predictions
        """
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            return self.model(data) 