"""
BatchNorm Visualization - Simplified Version with English Labels
================================================================
Professional data visualization for understanding BatchNorm effects
"""
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Configure matplotlib for better visualization
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

print("=== BatchNorm Comprehensive Visualization ===\n")

# 1. Basic BatchNorm functionality test
print("1. Basic BatchNorm Test:")
bn = nn.BatchNorm2d(6)
x = torch.randn(4, 6, 10, 10)
out = bn(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")

# 2. Effect of batch size on stability
print("\n2. Batch Size Effect Analysis:")

def test_batch_size_effect(batch_size):
    bn = nn.BatchNorm2d(1)
    x = torch.randn(batch_size, 1, 32, 32)
    with torch.no_grad():
        out = bn(x)
    return out.std().item()

batch_sizes = [1, 2, 4, 8, 16, 32]
stds = [test_batch_size_effect(bs) for bs in batch_sizes]

# Create comprehensive batch size analysis
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 1. Line plot showing batch size effect
ax1.plot(batch_sizes, stds, 'o-', linewidth=3, markersize=10, 
         markerfacecolor='coral', markeredgecolor='darkred', markeredgewidth=2)
ax1.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
ax1.set_ylabel('Output Standard Deviation', fontsize=14, fontweight='bold')
ax1.set_title('BatchNorm Stability vs Batch Size', fontsize=16, fontweight='bold', pad=20)
ax1.grid(True, alpha=0.3)

# Add value annotations
for i, (bs, std) in enumerate(zip(batch_sizes, stds)):
    ax1.annotate(f'{std:.3f}', (bs, std), textcoords="offset points", 
                xytext=(0,15), ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Add ideal line
ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2)
ax1.text(16, 1.05, 'Ideal Std = 1.0', fontsize=12, color='red', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Add explanation box
textstr = '''Key Insights:
• Larger batch size → More stable statistics
• Small batches → Unreliable normalization
• Recommended: batch_size ≥ 8
• Very small batches may hurt performance'''
props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=props)

# 2. Bar chart with stability regions
colors = ['red' if bs < 4 else 'orange' if bs < 8 else 'green' for bs in batch_sizes]
bars = ax2.bar(range(len(batch_sizes)), stds, color=colors, alpha=0.7, 
               edgecolor='black', linewidth=1, width=0.6)
ax2.set_xlabel('Batch Size', fontsize=14, fontweight='bold')
ax2.set_ylabel('Output Standard Deviation', fontsize=14, fontweight='bold')
ax2.set_title('Stability Analysis by Batch Size', fontsize=16, fontweight='bold', pad=20)
ax2.set_xticks(range(len(batch_sizes)))
ax2.set_xticklabels(batch_sizes)

# Add value labels on bars
for i, (bar, std) in enumerate(zip(bars, stds)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{std:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Add stability regions
ax2.axhline(y=1.1, color='red', linestyle=':', alpha=0.7, linewidth=2)
ax2.axhline(y=1.05, color='orange', linestyle=':', alpha=0.7, linewidth=2)
ax2.text(4.5, 1.12, 'Unstable Region', color='red', fontweight='bold', fontsize=12)
ax2.text(4.5, 1.07, 'Moderately Stable', color='orange', fontweight='bold', fontsize=12)
ax2.text(4.5, 1.02, 'Stable Region', color='green', fontweight='bold', fontsize=12)

# 3. Distribution comparison (before vs after BatchNorm)
print("\n3. Distribution Analysis:")
original_data = torch.randn(1000, 1, 1, 1)
bn_dist = nn.BatchNorm2d(1)
with torch.no_grad():
    normalized_data = bn_dist(original_data)

original_flat = original_data.numpy().flatten()
normalized_flat = normalized_data.numpy().flatten()

# Histogram comparison
ax3.hist(original_flat, bins=50, alpha=0.6, color='red', label='Original Data', density=True)
ax3.hist(normalized_flat, bins=50, alpha=0.6, color='blue', label='After BatchNorm', density=True)
ax3.set_xlabel('Value', fontsize=14, fontweight='bold')
ax3.set_ylabel('Density', fontsize=14, fontweight='bold')
ax3.set_title('Data Distribution: Before vs After BatchNorm', fontsize=16, fontweight='bold', pad=20)
ax3.legend(fontsize=12)
ax3.grid(True, alpha=0.3)

# Add statistics
mean_orig, std_orig = original_flat.mean(), original_flat.std()
mean_norm, std_norm = normalized_flat.mean(), normalized_flat.std()
ax3.axvline(mean_orig, color='red', linestyle='--', linewidth=2, alpha=0.7)
ax3.axvline(mean_norm, color='blue', linestyle='--', linewidth=2, alpha=0.7)

# 4. BatchNorm advantages summary
ax4.text(0.5, 0.95, 'BatchNorm Key Benefits', transform=ax4.transAxes, 
         fontsize=18, fontweight='bold', ha='center')

advantages = [
    '✓ Reduces Internal Covariate Shift',
    '✓ Enables Higher Learning Rates',
    '✓ Reduces Sensitivity to Weight Initialization', 
    '✓ Acts as Regularization (reduces overfitting)',
    '✓ Accelerates Training Convergence',
    '✓ Improves Gradient Flow in Deep Networks'
]

for i, advantage in enumerate(advantages):
    ax4.text(0.1, 0.8 - i*0.1, advantage, transform=ax4.transAxes, 
             fontsize=14, fontweight='bold', color='darkgreen')

# Best practices box
practices = '''Best Practices:
• Place after linear/conv layers, before activation
• Use model.train() during training
• Use model.eval() during inference
• Consider alternatives for small batches:
  - GroupNorm for computer vision
  - LayerNorm for NLP tasks
• Monitor running statistics during training'''

ax4.text(0.1, 0.35, practices, transform=ax4.transAxes, fontsize=12,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
         verticalalignment='top')

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.axis('off')

# Add overall statistics
stats_text = f'''Statistical Comparison:
                 Original    BatchNorm
Mean:            {mean_orig:7.3f}     {mean_norm:7.3f}
Std:             {std_orig:7.3f}     {std_norm:7.3f}
Min:             {original_flat.min():7.3f}     {normalized_flat.min():7.3f}
Max:             {original_flat.max():7.3f}     {normalized_flat.max():7.3f}'''

fig.text(0.02, 0.02, stats_text, fontsize=11, fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()

# 4. Training vs Evaluation mode analysis
print("\n4. Training vs Evaluation Mode Analysis:")

bn_mode = nn.BatchNorm2d(1)
torch.manual_seed(42)
test_batches = [torch.randn(8, 1, 4, 4) for _ in range(5)]

train_outputs = []
eval_outputs = []

# Training mode
bn_mode.train()
for batch in test_batches:
    out = bn_mode(batch)
    train_outputs.append((out.mean().item(), out.std().item()))

# Evaluation mode
bn_mode.eval()
for batch in test_batches:
    out = bn_mode(batch)
    eval_outputs.append((out.mean().item(), out.std().item()))

# Visualize mode differences
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

batch_nums = range(1, len(train_outputs) + 1)
train_means = [x[0] for x in train_outputs]
train_stds = [x[1] for x in train_outputs]
eval_means = [x[0] for x in eval_outputs]
eval_stds = [x[1] for x in eval_outputs]

# Mean comparison
ax1.plot(batch_nums, train_means, 'o-', label='Training Mode', linewidth=2, markersize=8, color='blue')
ax1.plot(batch_nums, eval_means, 's-', label='Evaluation Mode', linewidth=2, markersize=8, color='red')
ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax1.set_xlabel('Batch Number', fontsize=12, fontweight='bold')
ax1.set_ylabel('Output Mean', fontsize=12, fontweight='bold')
ax1.set_title('Training vs Evaluation Mode: Output Mean', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Std comparison
ax2.plot(batch_nums, train_stds, 'o-', label='Training Mode', linewidth=2, markersize=8, color='blue')
ax2.plot(batch_nums, eval_stds, 's-', label='Evaluation Mode', linewidth=2, markersize=8, color='red')
ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
ax2.set_xlabel('Batch Number', fontsize=12, fontweight='bold')
ax2.set_ylabel('Output Standard Deviation', fontsize=12, fontweight='bold')
ax2.set_title('Training vs Evaluation Mode: Output Std', fontsize=14, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== Analysis Complete ===")
print("Training mode: Uses batch statistics, updates running averages")
print("Evaluation mode: Uses fixed running statistics, ensures consistent output")
print("Key takeaway: Always switch modes appropriately!") 