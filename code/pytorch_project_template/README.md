# PyTorch Project Template

This template provides a structured approach to PyTorch projects with corresponding prompts for different scenarios.

## Project Structure

```
pytorch_project/
│
├── data/
│   ├── raw/              # Raw data
│   └── processed/        # Processed data
│
├── experiments/          # Experiment results and logs
│
├── src/
│   ├── data/            # Data loading and processing
│   ├── models/          # Model definitions
│   ├── training/        # Training loops and utilities
│   ├── evaluation/      # Evaluation metrics and utilities
│   └── utils/           # Helper functions
│
└── scripts/             # Training and evaluation scripts
```

## Prompt Templates

### 1. Data Processing Prompts

#### Data Loading
```python
"""
Create a PyTorch DataLoader for [specific data type] with the following requirements:
- Data location: [path]
- Batch size: [size]
- Required transformations: [list]
- Additional requirements: [list]
"""
```

#### Data Augmentation
```python
"""
Implement data augmentation pipeline for [data type] with:
- Required augmentations: [list]
- Probability for each augmentation: [list]
- Additional requirements: [list]
"""
```

### 2. Model Building Prompts

#### Basic Model
```python
"""
Create a PyTorch model for [task] with:
- Architecture: [type]
- Input shape: [shape]
- Output shape: [shape]
- Additional requirements: [list]
"""
```

#### Custom Layer
```python
"""
Implement a custom layer with:
- Layer type: [type]
- Input requirements: [list]
- Output requirements: [list]
- Additional features: [list]
"""
```

### 3. Training Prompts

#### Training Loop
```python
"""
Implement a training loop with:
- Model: [model]
- Optimizer: [type]
- Loss function: [type]
- Additional requirements: [list]
"""
```

#### Learning Rate Scheduling
```python
"""
Create a learning rate scheduler with:
- Initial learning rate: [value]
- Scheduler type: [type]
- Additional requirements: [list]
"""
```

### 4. Evaluation Prompts

#### Metrics Calculation
```python
"""
Implement evaluation metrics for [task] including:
- Required metrics: [list]
- Calculation method: [method]
- Additional requirements: [list]
"""
```

#### Model Evaluation
```python
"""
Create an evaluation pipeline with:
- Model: [model]
- Dataset: [dataset]
- Metrics: [list]
- Additional requirements: [list]
"""
```

### 5. Utility Prompts

#### Logging
```python
"""
Implement logging functionality with:
- Log types: [list]
- Output format: [format]
- Additional requirements: [list]
"""
```

#### Visualization
```python
"""
Create visualization utilities for:
- Data type: [type]
- Visualization type: [type]
- Additional requirements: [list]
"""
```

## Usage Guidelines

1. **Task Decomposition**
   - Break down complex tasks into smaller, manageable components
   - Use appropriate prompts for each component
   - Maintain consistency across components

2. **Prompt Customization**
   - Fill in the placeholders with specific requirements
   - Add or remove requirements as needed
   - Include relevant context and constraints

3. **Code Generation**
   - Use prompts with Cursor to generate code
   - Review and modify generated code as needed
   - Ensure code follows project structure

4. **Best Practices**
   - Keep code modular and reusable
   - Document code thoroughly
   - Follow PyTorch best practices
   - Implement proper error handling
   - Include unit tests where appropriate

## Example Usage

### Creating a DataLoader
```python
# Prompt
"""
Create a PyTorch DataLoader for image classification with the following requirements:
- Data location: ./data/raw/images
- Batch size: 32
- Required transformations: resize, normalize, random horizontal flip
- Additional requirements: shuffle data, use 4 workers
"""

# Generated code will be placed in src/data/dataloader.py
```

### Implementing a Model
```python
# Prompt
"""
Create a PyTorch model for image classification with:
- Architecture: ResNet18
- Input shape: (3, 224, 224)
- Output shape: (num_classes,)
- Additional requirements: pretrained weights, dropout
"""

# Generated code will be placed in src/models/resnet.py
```

## Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Best Practices for PyTorch Projects](https://pytorch.org/docs/stable/notes/coding_style.html) 