"""
Training script for PyTorch projects.
This script provides a standardized training pipeline.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any

from src.data.dataloader import get_dataloader, get_transforms
from src.models.base_model import ModelFactory
from src.training.trainer import Trainer
from src.utils.logger import ExperimentLogger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing the data')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name of the model to train')
    parser.add_argument('--pretrained', action='store_true',
                      help='Use pretrained weights')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                      help='Weight decay')
    
    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='experiments',
                      help='Directory to save logs')
    parser.add_argument('--experiment_name', type=str,
                      help='Name of the experiment')
    
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
            'pretrained': args.pretrained
        },
        'training': {
            'num_epochs': args.num_epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay
        },
        'logging': {
            'log_dir': args.log_dir,
            'experiment_name': args.experiment_name
        }
    }

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    config = get_config(args)
    
    # Setup logging
    logger = ExperimentLogger(
        base_dir=config['logging']['log_dir'],
        experiment_name=config['logging']['experiment_name']
    )
    logger.log_config(config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log_info(f'Using device: {device}')
    
    # Create data loaders
    train_transform = get_transforms(train=True)
    val_transform = get_transforms(train=False)
    
    train_loader = get_dataloader(
        data_dir=os.path.join(config['data']['data_dir'], 'train'),
        transform=train_transform,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    
    val_loader = get_dataloader(
        data_dir=os.path.join(config['data']['data_dir'], 'val'),
        transform=val_transform,
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )
    
    # Create model
    model = ModelFactory.create_model(
        model_name=config['model']['name'],
        config=config['model']
    )
    logger.log_model_summary(str(model))
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )
    
    # Train model
    try:
        trainer.train(
            num_epochs=config['training']['num_epochs'],
            save_dir=os.path.join(config['logging']['log_dir'],
                                config['logging']['experiment_name'])
        )
    except Exception as e:
        logger.log_error(e)
        raise

if __name__ == '__main__':
    main() 