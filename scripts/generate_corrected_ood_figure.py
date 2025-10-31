"""Generate corrected OOD comparison figure matching Table 2 data.

This script creates the exact figure shown in the paper with corrected values
from Table 2: NLP Domain Shift Results on AG News.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Data from Table 2 (CORRECTED VALUES)
configs = ['Baseline', 'APS-T', 'APS-C', 'APS-TC', 'APS-Full']
train_acc = [72.50, 72.50, 72.50, 72.50, 44.13]  # Training accuracy
ood_acc = [54.84, 54.84, 54.84, 54.84, 54.95]    # OOD accuracy (CORRECTED)
gaps = [17.66, 17.66, 17.66, 17.66, -10.82]      # Train-OOD gap (CORRECTED)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Left plot: Training vs OOD Accuracy ---
x = np.arange(len(configs))
width = 0.35

bars1 = ax1.bar(x - width/2, train_acc, width, label='Train Acc', 
                color='#5A7FA5', alpha=0.85)
bars2 = ax1.bar(x + width/2, ood_acc, width, label='OOD Acc', 
                color='#A05A7F', alpha=0.85)

ax1.set_ylabel('Accuracy (%)', fontsize=11)
ax1.set_title('Training vs OOD Accuracy Across Configurations', fontsize=12, pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(configs, rotation=0)
ax1.legend(loc='upper right', framealpha=0.95)
ax1.set_ylim(0, 80)
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9)

# --- Right plot: Train-OOD Gap ---
colors = ['#E08080' if g > 0 else '#80E080' for g in gaps]
bars3 = ax2.bar(configs, gaps, color=colors, alpha=0.85)

ax2.set_ylabel('Generalization Gap (pp)', fontsize=11)
ax2.set_title('Train-OOD Gap (lower is better)', fontsize=12, pad=15)
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.set_xticklabels(configs, rotation=0)
ax2.set_ylim(-15, 25)
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, gap in zip(bars3, gaps):
    height = bar.get_height()
    label_y = height + 1 if height > 0 else height - 1.5
    va = 'bottom' if height > 0 else 'top'
    ax2.text(bar.get_x() + bar.get_width()/2., label_y,
            f'{gap:.1f}' if height < 0 else f'{height:.1f}',
            ha='center', va=va, fontsize=9, fontweight='bold' if abs(height) < 1 else 'normal')

# Adjust layout and save
plt.tight_layout()

# Save to paper figures directory
output_path = Path(__file__).parent.parent / 'paper' / 'figures' / 'phase006b_ood_comparison.png'
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Saved corrected figure to: {output_path}")

# Also save to outputs for reference
output_path2 = Path(__file__).parent.parent / 'outputs' / 'phase006b_ood_comparison_corrected.png'
output_path2.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Backup saved to: {output_path2}")

plt.close()

# Print verification
print("\n" + "="*60)
print("Data Verification:")
print("="*60)
print(f"{'Config':<12} {'Train Acc':<12} {'OOD Acc':<12} {'Gap':<12}")
print("-"*60)
for i, config in enumerate(configs):
    print(f"{config:<12} {train_acc[i]:<12.2f} {ood_acc[i]:<12.2f} {gaps[i]:<12.2f}")
print("="*60)
print("\nKey findings:")
print("• Baseline through APS-TC: Train 72.50%, OOD 54.84%")
print("• APS-Full: Train 44.13%, OOD 54.95% (BEST OOD, negative gap)")
print("• APS-Full achieves -10.82pp gap (generalizes better than memorizes)")
