"""
Analyze T-C Conflict Experiment Results
========================================

This script analyzes the results from the T-C conflict synthetic experiment,
exploring the trade-off between topology preservation (T) and causal invariance (C).
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from scipy.interpolate import griddata

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_sweep_results(results_file: Path):
    """Load sweep results from JSON file."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    records = []
    for config_name, metrics in data.items():
        # Parse lambda values from config name (e.g., "T0.5_C1.0")
        parts = config_name.split('_')
        lambda_T = float(parts[0][1:])  # Remove 'T' prefix
        lambda_C = float(parts[1][1:])  # Remove 'C' prefix
        
        record = {
            'lambda_T': lambda_T,
            'lambda_C': lambda_C,
            'test_acc': metrics.get('test_acc', metrics.get('causal_accuracy', 0)),
            'causal_acc': metrics.get('causal_acc', metrics.get('causal_accuracy', 0)),
            'topo_preservation': metrics.get('topo_preservation', 0),
            'color_reliance': metrics.get('color_reliance', 0),
            'silhouette': metrics.get('silhouette', metrics.get('cluster_quality', 0)),
        }
        records.append(record)
    
    return pd.DataFrame(records)

def plot_pareto_frontier(df, output_dir):
    """Plot Pareto frontier showing T-C trade-off."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot colored by lambda values
    scatter = ax.scatter(df['topo_preservation'], df['causal_acc'], 
                         c=df['lambda_T'], s=150, cmap='viridis',
                         edgecolors='black', linewidth=1.5, alpha=0.8)
    
    # Identify Pareto-optimal points (simple heuristic: high on both metrics)
    pareto_threshold = 0.7
    pareto_mask = (df['topo_preservation'] >= pareto_threshold) & (df['causal_acc'] >= pareto_threshold)
    pareto_points = df[pareto_mask]
    
    if len(pareto_points) > 0:
        ax.scatter(pareto_points['topo_preservation'], pareto_points['causal_acc'],
                   s=200, facecolors='none', edgecolors='red', linewidth=3,
                   label='Pareto-Optimal', zorder=10)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('λ_T (Topology Weight)')
    
    ax.set_xlabel('Topology Preservation (kNN Jaccard)', fontsize=12)
    ax.set_ylabel('Causal Accuracy (Ignoring Spurious Color)', fontsize=12)
    ax.set_title('Topology-Causality Trade-off Analysis', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tc_pareto_frontier.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'tc_pareto_frontier.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'tc_pareto_frontier.png'}")
    plt.close()

def plot_heatmaps(df, output_dir):
    """Plot heatmaps for key metrics across lambda_T and lambda_C."""
    # Create pivot tables
    lambda_T_vals = sorted(df['lambda_T'].unique())
    lambda_C_vals = sorted(df['lambda_C'].unique())
    
    metrics = [
        ('causal_acc', 'Causal Accuracy'),
        ('topo_preservation', 'Topology Preservation'),
        ('color_reliance', 'Spurious Color Reliance'),
        ('test_acc', 'Test Accuracy')
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx]
        
        # Create pivot table
        pivot = df.pivot_table(values=metric, index='lambda_C', columns='lambda_T')
        
        # Plot heatmap
        cmap = 'RdYlGn' if metric != 'color_reliance' else 'RdYlGn_r'
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap, 
                    ax=ax, cbar_kws={'label': title},
                    vmin=0, vmax=1)
        
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel('λ_T (Topology Weight)', fontsize=11)
        ax.set_ylabel('λ_C (Causality Weight)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tc_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'tc_heatmaps.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'tc_heatmaps.png'}")
    plt.close()

def plot_lambda_sweep(df, output_dir):
    """Plot metric curves as functions of lambda weights."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Fix lambda_C=0, vary lambda_T (pure topology)
    ax = axes[0, 0]
    df_T = df[df['lambda_C'] == 0].sort_values('lambda_T')
    ax.plot(df_T['lambda_T'], df_T['causal_acc'], 'o-', label='Causal Acc', linewidth=2, markersize=8)
    ax.plot(df_T['lambda_T'], df_T['topo_preservation'], 's-', label='Topo Pres', linewidth=2, markersize=8)
    ax.plot(df_T['lambda_T'], df_T['test_acc'], '^-', label='Test Acc', linewidth=2, markersize=8)
    ax.set_xlabel('λ_T (λ_C = 0)', fontsize=11)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title('Pure Topology Effect', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Plot 2: Fix lambda_T=0, vary lambda_C (pure causality)
    ax = axes[0, 1]
    df_C = df[df['lambda_T'] == 0].sort_values('lambda_C')
    ax.plot(df_C['lambda_C'], df_C['causal_acc'], 'o-', label='Causal Acc', linewidth=2, markersize=8)
    ax.plot(df_C['lambda_C'], df_C['topo_preservation'], 's-', label='Topo Pres', linewidth=2, markersize=8)
    ax.plot(df_C['lambda_C'], df_C['test_acc'], '^-', label='Test Acc', linewidth=2, markersize=8)
    ax.set_xlabel('λ_C (λ_T = 0)', fontsize=11)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title('Pure Causality Effect', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Plot 3: Fix lambda_T=1, vary lambda_C (causality with topology)
    ax = axes[1, 0]
    df_TC = df[df['lambda_T'] == 1.0].sort_values('lambda_C')
    ax.plot(df_TC['lambda_C'], df_TC['causal_acc'], 'o-', label='Causal Acc', linewidth=2, markersize=8)
    ax.plot(df_TC['lambda_C'], df_TC['topo_preservation'], 's-', label='Topo Pres', linewidth=2, markersize=8)
    ax.plot(df_TC['lambda_C'], df_TC['test_acc'], '^-', label='Test Acc', linewidth=2, markersize=8)
    ax.set_xlabel('λ_C (λ_T = 1.0)', fontsize=11)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title('Causality with Fixed Topology', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Plot 4: Fix lambda_C=1, vary lambda_T (topology with causality)
    ax = axes[1, 1]
    df_CT = df[df['lambda_C'] == 1.0].sort_values('lambda_T')
    ax.plot(df_CT['lambda_T'], df_CT['causal_acc'], 'o-', label='Causal Acc', linewidth=2, markersize=8)
    ax.plot(df_CT['lambda_T'], df_CT['topo_preservation'], 's-', label='Topo Pres', linewidth=2, markersize=8)
    ax.plot(df_CT['lambda_T'], df_CT['test_acc'], '^-', label='Test Acc', linewidth=2, markersize=8)
    ax.set_xlabel('λ_T (λ_C = 1.0)', fontsize=11)
    ax.set_ylabel('Metric Value', fontsize=11)
    ax.set_title('Topology with Fixed Causality', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tc_lambda_sweeps.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'tc_lambda_sweeps.pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'tc_lambda_sweeps.png'}")
    plt.close()

def plot_3d_surface(df, output_dir):
    """Plot 3D surface of test accuracy over lambda_T and lambda_C space."""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    lambda_T = df['lambda_T'].values
    lambda_C = df['lambda_C'].values
    test_acc = df['test_acc'].values
    
    # Create regular grid for interpolation
    lambda_T_grid = np.linspace(lambda_T.min(), lambda_T.max(), 50)
    lambda_C_grid = np.linspace(lambda_C.min(), lambda_C.max(), 50)
    lambda_T_mesh, lambda_C_mesh = np.meshgrid(lambda_T_grid, lambda_C_grid)
    
    # Interpolate
    test_acc_mesh = griddata((lambda_T, lambda_C), test_acc,
                              (lambda_T_mesh, lambda_C_mesh), method='cubic')
    
    # Plot surface
    surf = ax.plot_surface(lambda_T_mesh, lambda_C_mesh, test_acc_mesh,
                           cmap='viridis', alpha=0.8, edgecolor='none')
    
    # Add scatter points
    ax.scatter(lambda_T, lambda_C, test_acc, c='red', s=50, alpha=0.6,
               edgecolors='black', linewidth=1)
    
    ax.set_xlabel('λ_T (Topology)', fontsize=11, labelpad=10)
    ax.set_ylabel('λ_C (Causality)', fontsize=11, labelpad=10)
    ax.set_zlabel('Test Accuracy', fontsize=11, labelpad=10)
    ax.set_title('Test Accuracy Landscape', fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tc_3d_surface.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'tc_3d_surface.png'}")
    plt.close()

def generate_latex_table(df, output_dir):
    """Generate LaTeX table showing representative configurations."""
    # Select interesting configurations
    configs = [
        (0.0, 0.0, 'Baseline'),
        (1.0, 0.0, 'T-only'),
        (0.0, 1.0, 'C-only'),
        (1.0, 1.0, 'T+C Balanced'),
        (0.5, 2.0, 'C-Dominant'),
        (2.0, 0.5, 'T-Dominant'),
    ]
    
    latex_lines = []
    latex_lines.append(r"\begin{table}[t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{T-C Conflict Experiment: Key Configurations}")
    latex_lines.append(r"\label{tab:tc_conflict}")
    latex_lines.append(r"\begin{tabular}{llcccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Configuration & ($\lambda_T$, $\lambda_C$) & Test Acc & Causal Acc & Topo Pres & Color Rel \\")
    latex_lines.append(r"\midrule")
    
    for lambda_T, lambda_C, name in configs:
        row = df[(df['lambda_T'] == lambda_T) & (df['lambda_C'] == lambda_C)]
        if len(row) > 0:
            r = row.iloc[0]
            latex_lines.append(
                f"{name} & ({lambda_T:.1f}, {lambda_C:.1f}) & "
                f"{r['test_acc']:.3f} & {r['causal_acc']:.3f} & "
                f"{r['topo_preservation']:.3f} & {r['color_reliance']:.3f} \\\\"
            )
    
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table}")
    
    latex_table = "\n".join(latex_lines)
    
    with open(output_dir / 'tc_conflict_table.tex', 'w') as f:
        f.write(latex_table)
    
    print(f"Saved: {output_dir / 'tc_conflict_table.tex'}")
    return latex_table

def compute_statistical_summary(df):
    """Compute key statistics and findings."""
    baseline = df[(df['lambda_T'] == 0) & (df['lambda_C'] == 0)].iloc[0]
    best_config = df.loc[df['test_acc'].idxmax()]
    best_causal = df.loc[df['causal_acc'].idxmax()]
    best_topo = df.loc[df['topo_preservation'].idxmax()]
    
    summary = {
        'baseline_test_acc': float(baseline['test_acc']),
        'baseline_causal_acc': float(baseline['causal_acc']),
        'baseline_topo': float(baseline['topo_preservation']),
        
        'best_config': (float(best_config['lambda_T']), float(best_config['lambda_C'])),
        'best_test_acc': float(best_config['test_acc']),
        'best_improvement': float(best_config['test_acc'] - baseline['test_acc']),
        
        'best_causal_config': (float(best_causal['lambda_T']), float(best_causal['lambda_C'])),
        'best_causal_acc': float(best_causal['causal_acc']),
        
        'best_topo_config': (float(best_topo['lambda_T']), float(best_topo['lambda_C'])),
        'best_topo_pres': float(best_topo['topo_preservation']),
        
        'mean_color_reliance': float(df['color_reliance'].mean()),
        'std_color_reliance': float(df['color_reliance'].std()),
    }
    
    return summary

def main():
    results_file = Path('outputs/tc_conflict/sweep_results.json')
    output_dir = Path('paper/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("T-C Conflict Experiment Analysis")
    print("=" * 60)
    
    # Load results
    print("\n1. Loading sweep results...")
    df = load_sweep_results(results_file)
    print(f"   Loaded {len(df)} configurations")
    print(f"   Lambda T range: {df['lambda_T'].min():.1f} - {df['lambda_T'].max():.1f}")
    print(f"   Lambda C range: {df['lambda_C'].min():.1f} - {df['lambda_C'].max():.1f}")
    
    # Compute statistical summary
    print("\n2. Statistical summary...")
    summary = compute_statistical_summary(df)
    print(f"   Baseline Test Acc: {summary['baseline_test_acc']:.3f}")
    print(f"   Best Config: λ_T={summary['best_config'][0]:.1f}, λ_C={summary['best_config'][1]:.1f}")
    print(f"   Best Test Acc: {summary['best_test_acc']:.3f}")
    print(f"   Improvement: {summary['best_improvement']:+.3f}")
    print(f"   Best Causal Acc: {summary['best_causal_acc']:.3f} at λ_T={summary['best_causal_config'][0]:.1f}, λ_C={summary['best_causal_config'][1]:.1f}")
    print(f"   Best Topo Pres: {summary['best_topo_pres']:.3f} at λ_T={summary['best_topo_config'][0]:.1f}, λ_C={summary['best_topo_config'][1]:.1f}")
    
    # Save summary
    with open(output_dir / 'tc_conflict_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate plots
    print("\n3. Generating visualizations...")
    plot_pareto_frontier(df, output_dir)
    plot_heatmaps(df, output_dir)
    plot_lambda_sweep(df, output_dir)
    plot_3d_surface(df, output_dir)
    
    # Generate LaTeX table
    print("\n4. Generating LaTeX table...")
    latex_table = generate_latex_table(df, output_dir)
    print("\nLaTeX Table Preview:")
    print(latex_table)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()
