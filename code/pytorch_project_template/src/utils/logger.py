"""
Logging utilities for PyTorch projects.
This module provides standardized logging functionality.
"""

import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime
import json

class Logger:
    """Standardized logger for PyTorch projects."""
    
    def __init__(self,
                 log_dir: str,
                 name: Optional[str] = None,
                 level: int = logging.INFO):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory to save logs
            name: Logger name
            level: Logging level
        """
        self.log_dir = log_dir
        self.name = name or 'pytorch_project'
        self.level = level
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logger
        self.setup_logger()
        
    def setup_logger(self):
        """Setup the logger configuration."""
        # Create logger
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)
        
        # Create handlers
        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, f'{self.name}.log')
        )
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
    def log_config(self, config: Dict[str, Any]):
        """
        Log configuration.
        
        Args:
            config: Configuration dictionary
        """
        config_path = os.path.join(self.log_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
            
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current step/epoch
        """
        metrics_str = ' - '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        self.logger.info(f'Step {step}: {metrics_str}')
        
    def log_error(self, error: Exception):
        """
        Log error.
        
        Args:
            error: Exception to log
        """
        self.logger.error(f'Error: {str(error)}', exc_info=True)
        
    def log_warning(self, message: str):
        """
        Log warning.
        
        Args:
            message: Warning message
        """
        self.logger.warning(message)
        
    def log_info(self, message: str):
        """
        Log info message.
        
        Args:
            message: Info message
        """
        self.logger.info(message)
        
    def log_debug(self, message: str):
        """
        Log debug message.
        
        Args:
            message: Debug message
        """
        self.logger.debug(message)
        
class ExperimentLogger(Logger):
    """Logger specifically for experiments."""
    
    def __init__(self,
                 base_dir: str,
                 experiment_name: Optional[str] = None,
                 **kwargs):
        """
        Initialize experiment logger.
        
        Args:
            base_dir: Base directory for experiments
            experiment_name: Name of the experiment
            **kwargs: Additional arguments for Logger
        """
        # Create experiment directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = experiment_name or f'experiment_{timestamp}'
        log_dir = os.path.join(base_dir, experiment_name)
        
        super().__init__(log_dir=log_dir, name=experiment_name, **kwargs)
        
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """
        Log hyperparameters.
        
        Args:
            hyperparameters: Dictionary of hyperparameter names and values
        """
        hyperparameters_path = os.path.join(self.log_dir, 'hyperparameters.json')
        with open(hyperparameters_path, 'w') as f:
            json.dump(hyperparameters, f, indent=4)
            
    def log_model_summary(self, model_summary: str):
        """
        Log model summary.
        
        Args:
            model_summary: Model architecture summary
        """
        summary_path = os.path.join(self.log_dir, 'model_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(model_summary) 