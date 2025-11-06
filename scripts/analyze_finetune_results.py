"""
Analyze NLP Fine-Tuning Experiment Results
==========================================

This script analyzes the results from Phase 006B (NLP fine-tuning) experiments,
comparing frozen vs fine-tuned BERT embeddings across APS component ablations.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_experiment_results(base_dir: Path):
    """Load results from all experiment variants."""
    results = {}
    
    variants = [
        'baseline_finetune',
        'aps-T_finetune',
        'aps-C_finetune',
        'aps-TC_finetune',
        'aps-full_finetune'
    ]
    
    for variant in variants:
        variant_dir = base_dir / variant
        metrics_file = variant_dir / 'final_metrics.json'
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                results[variant] = json.load(f)
        else:
            print(f"Warning: {metrics_file} not found")
    
    return results

def compare_with_frozen_baseline(finetune_results):
    """Compare fine-tuned results with frozen baseline from Phase 006A."""
    # Frozen baseline results (from previous experiments)
    frozen_baseline = {
        'train_acc': 0.985,
        'val_acc': 0.965,
        'test_acc': 0.752,  # OOD domain
    }
    
    comparison_data = []
    
    # Add frozen baseline
    comparison_data.append({
        'Model': 'Baseline (Frozen)',
        'Train Acc': frozen_baseline['train_acc'],
        'Val Acc': frozen_baseline['val_acc'],
        'OOD Acc': frozen_baseline['test_acc'],
        'Type': 'Frozen Embeddings'
    })
    
    # Add fine-tuned results
    for variant, metrics in finetune_results.items():
        model_name = variant.replace('_finetune', '').replace('aps-', 'APS-').replace('baseline', 'Baseline')
        comparison_data.append({
            'Model': model_name,
            'Train Acc': metrics.get('train_acc', 0),
            'Val Acc': metrics.get('val_acc', 0),
            'OOD Acc': metrics.get('test_acc', 0),
            'Type': 'Fine-Tuned'
        })
    
    return pd.DataFrame(comparison_data)

def plot_accuracy_comparison(df, output_dir):
    """Plot comparison of accuracies across models and embedding types."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    metrics = ['Train Acc', 'Val Acc', 'OOD Acc']
    titles = ['Training Accuracy', 'Validation Accuracy', 'OOD Test Accuracy']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        # Separate frozen and fine-tuned
        frozen_data = df[df['Type'] == 'Frozen Embeddings']
        finetune_data = df[df['Type'] == 'Fine-Tuned']
        
        x_pos = np.arange(len(finetune_data))
        width = 0.35
        
        # Plot fine-tuned bars
        bars1 = ax.bar(x_pos, finetune_data[metric], width, 
                       label='Fine-Tuned', alpha=0.8, color='steelblue')
        
        # Add frozen baseline as horizontal line
        frozen_val = frozen_data[metric].values[0]
        ax.axhline(y=frozen_val, color='coral', linestyle='--', 
                   linewidth=2, label='Frozen Baseline')
        
        ax.set_ylabel('Accuracy')
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(finetune_data['Model'], rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.05])
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'accuracy_comparison.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'accuracy_comparison.png'}")
    plt.close()

def plot_component_ablation(df, output_dir):
    """Plot ablation study showing effect of each component."""
    finetune_data = df[df['Type'] == 'Fine-Tuned'].copy()
    
    # Order by model complexity
    model_order = ['Baseline', 'APS-T', 'APS-C', 'APS-TC', 'APS-full']
    finetune_data['Model'] = pd.Categorical(finetune_data['Model'], 
                                            categories=model_order, ordered=True)
    finetune_data = finetune_data.sort_values('Model')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x_pos = np.arange(len(finetune_data))
    width = 0.25
    
    # Plot three metrics side by side
    bars1 = ax.bar(x_pos - width, finetune_data['Train Acc'], width, 
                   label='Train Acc', alpha=0.8, color='#2ecc71')
    bars2 = ax.bar(x_pos, finetune_data['Val Acc'], width, 
                   label='Val Acc', alpha=0.8, color='#3498db')
    bars3 = ax.bar(x_pos + width, finetune_data['OOD Acc'], width, 
                   label='OOD Acc', alpha=0.8, color='#e74c3c')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Component Ablation Study (Fine-Tuned BERT)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(finetune_data['Model'], rotation=45, ha='right')
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'component_ablation.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'component_ablation.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'component_ablation.png'}")
    plt.close()

def plot_ood_improvement(df, output_dir):
    """Focus plot on OOD accuracy improvements."""
    frozen_baseline = df[df['Type'] == 'Frozen Embeddings']['OOD Acc'].values[0]
    finetune_data = df[df['Type'] == 'Fine-Tuned'].copy()
    
    finetune_data['OOD Improvement'] = finetune_data['OOD Acc'] - frozen_baseline
    finetune_data = finetune_data.sort_values('OOD Acc')
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['coral' if x < 0 else 'steelblue' for x in finetune_data['OOD Improvement']]
    bars = ax.barh(finetune_data['Model'], finetune_data['OOD Improvement'], 
                   color=colors, alpha=0.8)
    
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('OOD Accuracy Improvement over Frozen Baseline')
    ax.set_title('Fine-Tuning Impact on OOD Generalization')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, finetune_data['OOD Improvement'])):
        label = f'{val:+.3f}'
        x_pos = val + (0.005 if val > 0 else -0.005)
        ha = 'left' if val > 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, label,
               ha=ha, va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ood_improvement.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'ood_improvement.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'ood_improvement.png'}")
    plt.close()

def generate_latex_table(df, output_dir):
    """Generate LaTeX table for paper."""
    finetune_data = df[df['Type'] == 'Fine-Tuned'].copy()
    frozen_data = df[df['Type'] == 'Frozen Embeddings'].copy()
    
    # Combine for table
    table_data = pd.concat([frozen_data, finetune_data])
    
    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{AG News Fine-Tuning Results: Component Ablation Study}")
    latex_lines.append(r"\label{tab:finetune_results}")
    latex_lines.append(r"\begin{tabular}{lccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Model & Train Acc & Val Acc & OOD Test Acc \\")
    latex_lines.append(r"\midrule")
    
    # Add frozen baseline
    for _, row in frozen_data.iterrows():
        latex_lines.append(
            f"{row['Model']} & {row['Train Acc']:.3f} & {row['Val Acc']:.3f} & {row['OOD Acc']:.3f} \\\\"
        )
    
    latex_lines.append(r"\midrule")
    
    # Add fine-tuned results
    for _, row in finetune_data.iterrows():
        # Bold best OOD accuracy
        ood_str = f"{row['OOD Acc']:.3f}"
        if row['OOD Acc'] == finetune_data['OOD Acc'].max():
            ood_str = r"\textbf{" + ood_str + "}"
        
        latex_lines.append(
            f"{row['Model']} & {row['Train Acc']:.3f} & {row['Val Acc']:.3f} & {ood_str} \\\\"
        )
    
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")
    
    latex_table = "\n".join(latex_lines)
    
    with open(output_dir / 'finetune_results_table.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"Saved: {output_dir / 'finetune_results_table.tex'}")
    return latex_table

def compute_statistical_summary(df):
    """Compute statistical summary of results."""
    frozen_baseline = df[df['Type'] == 'Frozen Embeddings']['OOD Acc'].values[0]
    finetune_data = df[df['Type'] == 'Fine-Tuned']
    
    summary = {
        'frozen_baseline_ood': frozen_baseline,
        'best_finetune_model': finetune_data.loc[finetune_data['OOD Acc'].idxmax(), 'Model'],
        'best_finetune_ood': finetune_data['OOD Acc'].max(),
        'improvement_over_frozen': finetune_data['OOD Acc'].max() - frozen_baseline,
        'baseline_finetune_ood': finetune_data[finetune_data['Model'] == 'Baseline']['OOD Acc'].values[0],
        'mean_aps_ood': finetune_data[finetune_data['Model'].str.contains('APS')]['OOD Acc'].mean(),
    }
    
    return summary

def main():
    base_dir = Path('outputs/phase006b_finetune')
    output_dir = Path('paper/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("NLP Fine-Tuning Experiment Analysis")
    print("=" * 60)
    
    # Load results
    print("\n1. Loading experiment results...")
    results = load_experiment_results(base_dir)
    print(f"   Loaded {len(results)} experiment variants")
    
    # Compare with frozen baseline
    print("\n2. Comparing with frozen baseline...")
    df = compare_with_frozen_baseline(results)
    print(df.to_string(index=False))
    
    # Compute statistical summary
    print("\n3. Statistical summary...")
    summary = compute_statistical_summary(df)
    print(f"   Frozen Baseline OOD: {summary['frozen_baseline_ood']:.3f}")
    print(f"   Best Fine-Tune Model: {summary['best_finetune_model']}")
    print(f"   Best Fine-Tune OOD: {summary['best_finetune_ood']:.3f}")
    print(f"   Improvement: {summary['improvement_over_frozen']:+.3f}")
    print(f"   Baseline Fine-Tune OOD: {summary['baseline_finetune_ood']:.3f}")
    print(f"   Mean APS OOD: {summary['mean_aps_ood']:.3f}")
    
    # Save summary
    with open(output_dir / 'finetune_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate plots
    print("\n4. Generating visualizations...")
    plot_accuracy_comparison(df, output_dir)
    plot_component_ablation(df, output_dir)
    plot_ood_improvement(df, output_dir)
    
    # Generate LaTeX table
    print("\n5. Generating LaTeX table...")
    latex_table = generate_latex_table(df, output_dir)
    print("\nLaTeX Table Preview:")
    print(latex_table)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()
