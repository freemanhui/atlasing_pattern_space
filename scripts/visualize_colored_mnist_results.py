"""Visualize ColoredMNIST experiment results."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
exps = ['baseline', 'aps-t', 'aps-c', 'aps-tc']
results = []
for exp in exps:
    with open(f'outputs/colored_mnist/{exp}/results.json') as f:
        results.append(json.load(f))

# Extract metrics
names = [r['experiment'] for r in results]
test_accs = [r['best_test_accuracy'] for r in results]
causal_ratios = [r['causal_metrics']['causal_ratio'] for r in results]
reliance_gaps = [r['causal_metrics']['reliance_gap'] for r in results]
inv_scores = [r['causal_metrics']['invariance_score'] for r in results]

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
colors = ['steelblue', 'orange', 'green', 'purple']

# Test Accuracy
axes[0, 0].bar(range(len(names)), test_accs, color=colors)
axes[0, 0].set_xticks(range(len(names)))
axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
axes[0, 0].set_ylabel('Test Accuracy')
axes[0, 0].set_title('Test Accuracy (Higher = Better)')
axes[0, 0].set_ylim([0.96, 0.98])
axes[0, 0].grid(axis='y', alpha=0.3)
for i, (x, y) in enumerate(zip(range(len(names)), test_accs)):
    axes[0, 0].text(x, y + 0.0002, f'{y:.4f}', ha='center', va='bottom', fontsize=9)

# Causal Ratio
axes[0, 1].bar(range(len(names)), causal_ratios, color=colors)
axes[0, 1].set_xticks(range(len(names)))
axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
axes[0, 1].set_ylabel('Causal Ratio')
axes[0, 1].set_title('Causal Ratio: Label/Color Correlation (Higher = Better)')
axes[0, 1].grid(axis='y', alpha=0.3)
for i, (x, y) in enumerate(zip(range(len(names)), causal_ratios)):
    axes[0, 1].text(x, y + 0.2, f'{y:.2f}', ha='center', va='bottom', fontsize=9)

# Reliance Gap
axes[1, 0].bar(range(len(names)), reliance_gaps, color=colors)
axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
axes[1, 0].set_xticks(range(len(names)))
axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
axes[1, 0].set_ylabel('Reliance Gap')
axes[1, 0].set_title('Spurious Reliance Gap (Near 0 = Better)')
axes[1, 0].grid(axis='y', alpha=0.3)
for i, (x, y) in enumerate(zip(range(len(names)), reliance_gaps)):
    axes[1, 0].text(x, y - 0.003, f'{y:.4f}', ha='center', va='top', fontsize=9)

# Invariance Score
axes[1, 1].bar(range(len(names)), inv_scores, color=colors)
axes[1, 1].set_xticks(range(len(names)))
axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
axes[1, 1].set_ylabel('Invariance Score')
axes[1, 1].set_title('Environment Invariance (Higher = Better)')
axes[1, 1].set_ylim([0.98, 0.99])
axes[1, 1].grid(axis='y', alpha=0.3)
for i, (x, y) in enumerate(zip(range(len(names)), inv_scores)):
    axes[1, 1].text(x, y + 0.0001, f'{y:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/colored_mnist/comparison_working.png', dpi=150, bbox_inches='tight')
print('Saved to outputs/colored_mnist/comparison_working.png')

# Print table
print('\n' + '='*90)
print(f"{'Experiment':<15} {'Test Acc':<12} {'Causal Ratio':<15} {'Reliance Gap':<15} {'Invariance':<12}")
print('='*90)
for i, name in enumerate(names):
    print(f"{name:<15} {test_accs[i]:<12.4f} {causal_ratios[i]:<15.4f} {reliance_gaps[i]:<15.4f} {inv_scores[i]:<12.4f}")
print('='*90)
