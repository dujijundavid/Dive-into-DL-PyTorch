"""
Evaluation script for PyTorch projects.
This script provides a standardized evaluation pipeline.
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
from typing import Dict, Any
import json

from src.data.dataloader import get_dataloader, get_transforms
from src.models.base_model import ModelFactory
from src.evaluation.metrics import Metrics
from src.utils.logger import Logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate a model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing the test data')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name of the model to evaluate')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to model checkpoint')
    
    # Evaluation arguments
    parser.add_argument('--task_type', type=str, default='classification',
                      choices=['classification', 'regression'],
                      help='Type of task')
    
    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='evaluation_logs',
                      help='Directory to save evaluation logs')
    
    return parser.parse_args()

def get_config(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Create configuration dictionary from arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Configuration dictionary
    """
    return {
        'data': {
            'data_dir': args.data_dir,
            'batch_size': args.batch_size,
            'num_workers': args.num_workers
        },
        'model': {
            'name': args.model_name,
            'checkpoint_path': args.checkpoint_path
        },
        'evaluation': {
            'task_type': args.task_type
        },
        'logging': {
            'log_dir': args.log_dir
        }
    }

def main():
    """Main evaluation function."""
    # Parse arguments
    args = parse_args()
    config = get_config(args)
    
    # Setup logging
    logger = Logger(
        log_dir=config['logging']['log_dir'],
        name='evaluation'
    )
    logger.log_config(config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log_info(f'Using device: {device}')
    
    # Create data loader
    test_transform = get_transforms(train=False)
    test_loader = get_dataloader(
        data_dir=config['data']['data_dir'],
        transform=test_transform,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    
    # Create model
    model = ModelFactory.create_model(
        model_name=config['model']['name'],
        config=config['model']
    )
    
    # Load checkpoint
    model.load_checkpoint(config['model']['checkpoint_path'])
    model = model.to(device)
    model.eval()
    
    # Evaluate model
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            all_outputs.append(output)
            all_targets.append(target)
    
    # Concatenate outputs and targets
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate metrics
    metrics = Metrics.get_all_metrics(
        all_outputs,
        all_targets,
        task_type=config['evaluation']['task_type']
    )
    
    # Log metrics
    logger.log_info('Evaluation metrics:')
    for metric_name, metric_value in metrics.items():
        logger.log_info(f'{metric_name}: {metric_value:.4f}')
    
    # Save metrics
    metrics_path = os.path.join(config['logging']['log_dir'], 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

if __name__ == '__main__':
    main() 