"""
Generate all figures for the causal learning paper.

This script creates:
1. Performance curves across correlation spectrum
2. Causality metrics comparison
3. Latent space visualizations
4. Energy landscape plots (if available)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

# Output directory
OUTPUT_DIR = Path("paper/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load consolidated results
CONSOLIDATED_FILE = Path("outputs/consolidated_summary.json")

def load_all_results():
    """Load all consolidated results."""
    if not CONSOLIDATED_FILE.exists():
        print(f"Error: {CONSOLIDATED_FILE} not found. Run consolidate_results.py first.")
        return {}
    
    with open(CONSOLIDATED_FILE) as f:
        return json.load(f)

def load_results(version):
    """Load results for a specific version from consolidated data."""
    all_results = load_all_results()
    return all_results.get(version, None)


def create_performance_spectrum_plot():
    """Figure 1: Performance across correlation spectrum."""
    versions = ["v2", "v3.1"]  # Only versions with data
    version_labels = {
        "v2": "99.5% Corr\n(Hard)",
        "v3.1": "99%/-99% Corr\n(Very Hard)"
    }
    
    models = ["baseline", "APS-T", "APS-C", "APS-Full"]
    model_labels = {
        "baseline": "Baseline",
        "APS-T": "APS-T",
        "APS-C": "APS-C",
        "APS-Full": "APS-Full"
    }
    
    # Collect data
    data = {model: [] for model in models}
    
    for version in versions:
        results = load_results(version)
        if results is None:
            continue
        
        for model in models:
            if model in results:
                acc = results[model].get("test_accuracy", 0) * 100
                data[model].append(acc)
            else:
                data[model].append(0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    
    x = np.arange(len(versions))
    width = 0.2
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    for i, (model, label) in enumerate(model_labels.items()):
        offset = (i - 1.5) * width
        ax.bar(x + offset, data[model], width, label=label, color=colors[i], alpha=0.8)
    
    ax.set_xlabel('Correlation Strength', fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax.set_title('Performance Across Correlation Spectrum', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([version_labels[v] for v in versions])
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig1_performance_spectrum.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig1_performance_spectrum.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created Figure 1: Performance spectrum")


def create_causality_metrics_plot():
    """Figure 2: Causality metrics comparison."""
    versions = ["v2", "v3.1"]
    version_labels = {
        "v2": "99.5% Corr",
        "v3.1": "99%/-99% Corr"
    }
    
    models = ["baseline", "APS-C", "APS-Full"]
    model_labels = {
        "baseline": "Baseline",
        "APS-C": "APS-C",
        "APS-Full": "APS-Full"
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    
    # Metric 1: Causal Ratio
    ax = axes[0]
    data = {model: [] for model in models}
    
    for version in versions:
        results = load_results(version)
        if results is None:
            for model in models:
                data[model].append(0)
            continue
        
        for model in models:
            if model in results:
                ratio = results[model].get("causal_ratio", 0)
                data[model].append(ratio)
            else:
                data[model].append(0)
    
    x = np.arange(len(versions))
    width = 0.25
    colors = ['#2E86AB', '#F18F01', '#C73E1D']
    
    for i, (model, label) in enumerate(model_labels.items()):
        offset = (i - 1) * width
        ax.bar(x + offset, data[model], width, label=label, color=colors[i], alpha=0.8)
    
    ax.set_ylabel('Causal Ratio', fontweight='bold')
    ax.set_title('Causal Feature Reliance', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([version_labels[v] for v in versions], fontsize=8)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    
    # Metric 2: Spurious Reliance
    ax = axes[1]
    data = {model: [] for model in models}
    
    for version in versions:
        results = load_results(version)
        if results is None:
            for model in models:
                data[model].append(0)
            continue
        
        for model in models:
            if model in results:
                reliance = abs(results[model].get("reliance_gap", 0)) * 100
                data[model].append(reliance)
            else:
                data[model].append(0)
    
    for i, (model, label) in enumerate(model_labels.items()):
        offset = (i - 1) * width
        ax.bar(x + offset, data[model], width, label=label, color=colors[i], alpha=0.8)
    
    ax.set_ylabel('|Reliance Gap| (%)', fontweight='bold')
    ax.set_title('Spurious Feature Sensitivity', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([version_labels[v] for v in versions], fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    # Metric 3: Environment Invariance
    ax = axes[2]
    data = {model: [] for model in models}
    
    for version in versions:
        results = load_results(version)
        if results is None:
            for model in models:
                data[model].append(0)
            continue
        
        for model in models:
            if model in results:
                inv = results[model].get("env_invariance", 0)
                data[model].append(inv)
            else:
                data[model].append(0)
    
    for i, (model, label) in enumerate(model_labels.items()):
        offset = (i - 1) * width
        ax.bar(x + offset, data[model], width, label=label, color=colors[i], alpha=0.8)
    
    ax.set_ylabel('Invariance Score', fontweight='bold')
    ax.set_title('Environment Consistency', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([version_labels[v] for v in versions], fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig2_causality_metrics.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig2_causality_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created Figure 2: Causality metrics")


def create_phase_transition_plot():
    """Figure 3: Correlation strength vs performance."""
    versions = ["v2", "v3.1"]
    version_labels = {
        "v2": "99.5%",
        "v3.1": "99%"
    }
    
    models = ["baseline", "APS-C", "APS-Full"]
    model_labels = {
        "baseline": "Baseline",
        "APS-C": "APS-C",
        "APS-Full": "APS-Full"
    }
    
    # Collect data
    data = {model: [] for model in models}
    
    for version in versions:
        results = load_results(version)
        if results is None:
            for model in models:
                data[model].append(0)
            continue
        
        for model in models:
            if model in results:
                acc = results[model].get("test_accuracy", 0) * 100
                data[model].append(acc)
            else:
                data[model].append(0)
    
    # Plot
    fig, ax = plt.subplots(figsize=(7, 4))
    
    colors = ['#2E86AB', '#F18F01', '#C73E1D']
    markers = ['o', 's', '^']
    
    for i, (model, label) in enumerate(model_labels.items()):
        ax.plot(range(len(versions)), data[model], 
                marker=markers[i], color=colors[i], label=label,
                linewidth=2, markersize=8, alpha=0.8)
    
    ax.set_xlabel('Training Correlation Strength', fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontweight='bold')
    ax.set_title('Phase Transition at 100% Correlation', fontweight='bold')
    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels([version_labels[v] for v in versions])
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim([-5, 105])
    ax.grid(alpha=0.3)
    
    # Add annotation for phase transition
    ax.axvline(x=2, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(2.1, 50, 'Phase\nTransition', color='red', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig3_phase_transition.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig3_phase_transition.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created Figure 3: Phase transition")


def create_sample_visualizations():
    """Figure 4: Sample images from each environment."""
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    
    versions = ["v2", "v3.1"]
    version_titles = {
        "v2": "99.5% Correlation",
        "v3.1": "99%/-99% Correlation"
    }
    
    for row, version in enumerate(versions):
        # Create placeholders - in real paper, add actual sample images
        env_dirs = ["env0", "env1", "test"]
        for col, env in enumerate(env_dirs):
            ax = axes[row, col]
            
            # For now, just create placeholder colored boxes to show structure
            # In actual implementation, load real samples from saved data
            if env == "test":
                # Test environment has different correlation
                color = 'lightcoral' if version == "v3.1" else 'lightblue'
            else:
                color = 'lightgreen'
            
            ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.3))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')
            ax.axis('off')
            
            if row == 0:
                env_label = {"env0": "Env 0", "env1": "Env 1", "test": "Test"}
                ax.set_title(env_label[env], fontweight='bold')
            
            if col == 0:
                ax.text(-0.2, 0.5, version_titles[version], 
                       rotation=90, va='center', ha='right',
                       fontweight='bold', fontsize=10,
                       transform=ax.transAxes)
        
        # Add summary stats in 4th column
        ax = axes[row, 3]
        results = load_results(version)
        if results and "baseline" in results:
            acc = results["baseline"].get("test_accuracy", 0) * 100
            ratio = results["baseline"].get("causal_ratio", 0)
            
            text = "Baseline Results:\n\n"
            text += f"Test Acc: {acc:.1f}%\n"
            text += f"Causal Ratio: {ratio:.2f}"
            
            ax.text(0.1, 0.5, text, fontsize=9, va='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_sample_visualization.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig4_sample_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Created Figure 4: Sample visualization")


def create_ablation_study_table():
    """Generate LaTeX table for all ablation results."""
    versions = ["v2", "v3.1"]
    version_names = {
        "v2": "Hard (99.5\\%)",
        "v3.1": "Very Hard (99\\%/-99\\%)"
    }
    
    models = ["baseline", "APS-T", "APS-C", "APS-Full"]
    model_names = {
        "baseline": "Baseline",
        "APS-T": "$+$ Topology",
        "APS-C": "$+$ Causality",
        "APS-Full": "Full APS"
    }
    
    # Build table
    latex = "\\begin{table}[t]\n"
    latex += "\\centering\n"
    latex += "\\caption{Performance across correlation spectrum. Bold indicates best in column.}\n"
    latex += "\\label{tab:ablation}\n"
    latex += "\\small\n"
    latex += "\\begin{tabular}{lccccc}\n"
    latex += "\\toprule\n"
    latex += "& & \\multicolumn{4}{c}{\\textbf{Version}} \\\\\n"
    latex += "\\cmidrule{3-6}\n"
    latex += "\\textbf{Model} & \\textbf{Metric} & " + " & ".join([version_names[v] for v in versions]) + " \\\\\n"
    latex += "\\midrule\n"
    
    for model in models:
        # Accuracy row
        latex += model_names[model] + " & Acc (\\%) & "
        
        values = []
        for version in versions:
            results = load_results(version)
            if results and model in results:
                acc = results[model].get("test_accuracy", 0) * 100
                values.append(f"{acc:.2f}")
            else:
                values.append("--")
        
        latex += " & ".join(values) + " \\\\\n"
        
        # Causal ratio row
        latex += " & Causal Ratio & "
        values = []
        for version in versions:
            results = load_results(version)
            if results and model in results:
                ratio = results[model].get("causal_ratio", 0)
                values.append(f"{ratio:.2f}")
            else:
                values.append("--")
        
        latex += " & ".join(values) + " \\\\\n"
        
        if model != "APS-Full":
            latex += "\\cmidrule{2-6}\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    # Save to file
    table_file = OUTPUT_DIR.parent / "tables" / "tab1_ablation.tex"
    table_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(table_file, 'w') as f:
        f.write(latex)
    
    print(f"✓ Created Table 1: Ablation study → {table_file}")


def main():
    """Generate all figures and tables."""
    print("Generating paper figures...\n")
    
    create_performance_spectrum_plot()
    create_causality_metrics_plot()
    create_phase_transition_plot()
    create_sample_visualizations()
    create_ablation_study_table()
    
    print(f"\n✓ All figures saved to: {OUTPUT_DIR}")
    print(f"✓ Tables saved to: {OUTPUT_DIR.parent / 'tables'}")


if __name__ == "__main__":
    main()
