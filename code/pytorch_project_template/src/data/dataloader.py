"""
Data loading utilities for PyTorch projects.
This module provides standardized data loading functionality.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Optional, Tuple, List, Dict, Any

class BaseDataset(Dataset):
    """Base dataset class that implements common functionality."""
    
    def __init__(self, 
                 data_path: str,
                 transform: Optional[transforms.Compose] = None,
                 target_transform: Optional[transforms.Compose] = None):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the data
            transform: Optional transform to be applied on the data
            target_transform: Optional transform to be applied on the target
        """
        self.data_path = data_path
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self) -> int:
        """Return the total number of samples."""
        raise NotImplementedError
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a sample from the dataset."""
        raise NotImplementedError

def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs
) -> DataLoader:
    """
    Create a DataLoader with standard configurations.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory in CPU
        **kwargs: Additional arguments for DataLoader
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **kwargs
    )

def get_transforms(
    train: bool = True,
    input_size: Tuple[int, int] = (224, 224),
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> transforms.Compose:
    """
    Get standard transforms for training or validation.
    
    Args:
        train: Whether to get training transforms
        input_size: Input image size
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        Compose transform
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ]) 