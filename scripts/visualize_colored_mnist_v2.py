"""Visualize ColoredMNIST v2 experiment results (harder task)."""

import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
exps = ['baseline', 'aps-t', 'aps-c', 'aps-e', 'aps-tc', 'aps-full']
results = []
for exp in exps:
    try:
        with open(f'outputs/colored_mnist/{exp}/results.json') as f:
            results.append(json.load(f))
    except FileNotFoundError:
        print(f"Warning: {exp} results not found, skipping")

# Extract metrics
names = [r['experiment'] for r in results]
test_accs = [r['best_test_accuracy'] for r in results]
causal_ratios = [r['causal_metrics']['causal_ratio'] for r in results]
reliance_gaps = [r['causal_metrics']['reliance_gap'] for r in results]
inv_scores = [r['causal_metrics']['invariance_score'] for r in results]

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors = ['steelblue', 'orange', 'green', 'red', 'purple', 'brown'][:len(names)]

# Test Accuracy
axes[0, 0].bar(range(len(names)), test_accs, color=colors)
axes[0, 0].set_xticks(range(len(names)))
axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
axes[0, 0].set_ylabel('Test Accuracy', fontsize=12)
axes[0, 0].set_title('Test Accuracy on Harder Task (Higher = Better)', fontsize=13, fontweight='bold')
axes[0, 0].set_ylim([0.7, 0.85])
axes[0, 0].axhline(0.82, color='gray', linestyle='--', alpha=0.5, label='Target: 82%')
axes[0, 0].grid(axis='y', alpha=0.3)
axes[0, 0].legend()
for i, (x, y) in enumerate(zip(range(len(names)), test_accs)):
    axes[0, 0].text(x, y + 0.003, f'{y:.3f}', ha='center', va='bottom', fontsize=9)

# Causal Ratio
axes[0, 1].bar(range(len(names)), causal_ratios, color=colors)
axes[0, 1].set_xticks(range(len(names)))
axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
axes[0, 1].set_ylabel('Causal Ratio', fontsize=12)
axes[0, 1].set_title('Causal Ratio: Label/Color Correlation (Higher = Better)', fontsize=13, fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)
for i, (x, y) in enumerate(zip(range(len(names)), causal_ratios)):
    axes[0, 1].text(x, y + 0.05, f'{y:.2f}', ha='center', va='bottom', fontsize=9)

# Reliance Gap
axes[1, 0].bar(range(len(names)), reliance_gaps, color=colors)
axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1.5)
axes[1, 0].set_xticks(range(len(names)))
axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
axes[1, 0].set_ylabel('Reliance Gap', fontsize=12)
axes[1, 0].set_title('Spurious Reliance Gap (Near 0 = Better)', fontsize=13, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)
for i, (x, y) in enumerate(zip(range(len(names)), reliance_gaps)):
    axes[1, 0].text(x, y - 0.02, f'{y:.3f}', ha='center', va='top', fontsize=9)

# Invariance Score
axes[1, 1].bar(range(len(names)), inv_scores, color=colors)
axes[1, 1].set_xticks(range(len(names)))
axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
axes[1, 1].set_ylabel('Invariance Score', fontsize=12)
axes[1, 1].set_title('Environment Invariance (Higher = Better)', fontsize=13, fontweight='bold')
axes[1, 1].set_ylim([0.90, 0.92])
axes[1, 1].grid(axis='y', alpha=0.3)
for i, (x, y) in enumerate(zip(range(len(names)), inv_scores)):
    axes[1, 1].text(x, y + 0.0003, f'{y:.4f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('ColoredMNIST v2: Harder Task (99.5%/99% Train Corr, 5% Test Corr)', 
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('outputs/colored_mnist/comparison_v2_harder.png', dpi=150, bbox_inches='tight')
print('Saved to outputs/colored_mnist/comparison_v2_harder.png')

# Print table
print('\n' + '='*95)
print(f"{'Experiment':<12} {'Test Acc':<12} {'Causal Ratio':<15} {'Reliance Gap':<15} {'Invariance':<12}")
print('='*95)
for i, name in enumerate(names):
    print(f"{name:<12} {test_accs[i]:<12.4f} {causal_ratios[i]:<15.4f} {reliance_gaps[i]:<15.4f} {inv_scores[i]:<12.4f}")
print('='*95)

# Summary insights
print('\n' + '='*95)
print('KEY FINDINGS:')
print('='*95)
best_acc_idx = np.argmax(test_accs)
print(f"1. Best Test Accuracy: {names[best_acc_idx]} ({test_accs[best_acc_idx]:.4f})")

best_causal_idx = np.argmax(causal_ratios)
print(f"2. Best Causal Ratio: {names[best_causal_idx]} ({causal_ratios[best_causal_idx]:.4f})")

best_reliance_idx = np.argmin(np.abs(reliance_gaps))
print(f"3. Best Reliance Gap (closest to 0): {names[best_reliance_idx]} ({reliance_gaps[best_reliance_idx]:.4f})")

print("\n4. Task Difficulty: All models ~82% accuracy (vs 97% on easy task)")
print("5. Energy Component: Now stable with β=1.0, λ_E=0.01")
print('='*95)
