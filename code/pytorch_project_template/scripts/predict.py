"""
Prediction script for PyTorch projects.
This script provides a standardized prediction pipeline.
"""

import os
import argparse
import torch
import json
from typing import Dict, Any, List
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from src.models.base_model import ModelFactory
from src.utils.logger import Logger

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions')
    
    # Input arguments
    parser.add_argument('--input_path', type=str, required=True,
                      help='Path to input data (file or directory)')
    parser.add_argument('--input_type', type=str, default='image',
                      choices=['image', 'text', 'audio'],
                      help='Type of input data')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, required=True,
                      help='Name of the model to use')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                      help='Path to model checkpoint')
    
    # Output arguments
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path to save predictions')
    parser.add_argument('--output_format', type=str, default='json',
                      choices=['json', 'csv', 'txt'],
                      help='Format of output predictions')
    
    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='prediction_logs',
                      help='Directory to save prediction logs')
    
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
        'input': {
            'path': args.input_path,
            'type': args.input_type
        },
        'model': {
            'name': args.model_name,
            'checkpoint_path': args.checkpoint_path
        },
        'output': {
            'path': args.output_path,
            'format': args.output_format
        },
        'logging': {
            'log_dir': args.log_dir
        }
    }

def load_input_data(config: Dict[str, Any]) -> List[Any]:
    """
    Load input data based on type.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of input data
    """
    input_path = config['input']['path']
    input_type = config['input']['type']
    
    if input_type == 'image':
        if os.path.isfile(input_path):
            return [Image.open(input_path)]
        elif os.path.isdir(input_path):
            return [Image.open(os.path.join(input_path, f))
                   for f in os.listdir(input_path)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    else:
        raise NotImplementedError(f'Input type {input_type} not implemented')

def preprocess_data(data: List[Any], config: Dict[str, Any]) -> torch.Tensor:
    """
    Preprocess input data.
    
    Args:
        data: List of input data
        config: Configuration dictionary
        
    Returns:
        Preprocessed data tensor
    """
    input_type = config['input']['type']
    
    if input_type == 'image':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        return torch.stack([transform(img) for img in data])
    else:
        raise NotImplementedError(f'Input type {input_type} not implemented')

def save_predictions(predictions: List[Any], config: Dict[str, Any]):
    """
    Save predictions to file.
    
    Args:
        predictions: List of predictions
        config: Configuration dictionary
    """
    output_path = config['output']['path']
    output_format = config['output']['format']
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_format == 'json':
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=4)
    elif output_format == 'csv':
        import pandas as pd
        pd.DataFrame(predictions).to_csv(output_path, index=False)
    elif output_format == 'txt':
        with open(output_path, 'w') as f:
            for pred in predictions:
                f.write(f'{pred}\n')

def main():
    """Main prediction function."""
    # Parse arguments
    args = parse_args()
    config = get_config(args)
    
    # Setup logging
    logger = Logger(
        log_dir=config['logging']['log_dir'],
        name='prediction'
    )
    logger.log_config(config)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.log_info(f'Using device: {device}')
    
    # Create model
    model = ModelFactory.create_model(
        model_name=config['model']['name'],
        config=config['model']
    )
    
    # Load checkpoint
    model.load_checkpoint(config['model']['checkpoint_path'])
    model = model.to(device)
    model.eval()
    
    # Load and preprocess data
    input_data = load_input_data(config)
    preprocessed_data = preprocess_data(input_data, config)
    
    # Make predictions
    with torch.no_grad():
        predictions = model(preprocessed_data.to(device))
        predictions = predictions.cpu().numpy()
    
    # Save predictions
    save_predictions(predictions.tolist(), config)
    logger.log_info(f'Predictions saved to {config["output"]["path"]}')

if __name__ == '__main__':
    main() 