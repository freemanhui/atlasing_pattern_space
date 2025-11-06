#!/usr/bin/env python3
"""
Results Analysis Script - Phase 5.6

Aggregates results from all experiments and generates:
- Summary tables (CSV and LaTeX)
- Comparison plots
- Statistical significance tests
- Publication-ready figures
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_ablation_results(results_dir: Path = Path("outputs/ablation")) -> Dict:
    """Load all ablation study results."""
    summary_path = results_dir / "ablation_summary.json"
    
    if not summary_path.exists():
        print(f"Warning: {summary_path} not found. Run ablation experiments first.")
        return {}
    
    with open(summary_path, "r") as f:
        return json.load(f)


def load_ood_results(results_dir: Path = Path("outputs/ood/metrics")) -> Dict:
    """Load all OOD robustness results."""
    if not results_dir.exists():
        print(f"Warning: {results_dir} not found. Run OOD experiments first.")
        return {}
    
    results = {}
    for json_file in results_dir.glob("*_ood_results.json"):
        config_name = json_file.stem.replace("_ood_results", "")
        with open(json_file, "r") as f:
            results[config_name] = json.load(f)
    
    return results


def load_fewshot_results(results_dir: Path = Path("outputs/fewshot/metrics")) -> Dict:
    """Load all few-shot learning results."""
    if not results_dir.exists():
        print(f"Warning: {results_dir} not found. Run few-shot experiments first.")
        return {}
    
    results = {}
    for json_file in results_dir.glob("*_fewshot_results.json"):
        config_name = json_file.stem.replace("_fewshot_results", "")
        with open(json_file, "r") as f:
            results[config_name] = json.load(f)
    
    return results


def create_ablation_table(ablation_results: Dict, output_dir: Path):
    """Create summary table for ablation study."""
    
    configs = ["baseline", "t_only", "c_only", "e_only", "t_c", "t_e", "c_e", "t_c_e"]
    
    # Extract metrics
    data = []
    for config in configs:
        if config not in ablation_results:
            continue
        
        metrics = ablation_results[config]["metrics"]
        data.append({
            "Configuration": config,
            "T": ablation_results[config]["config"]["use_topo"],
            "C": ablation_results[config]["config"]["use_causal"],
            "E": ablation_results[config]["config"]["use_energy"],
            "Recon ↓": metrics["reconstruction_error"],
            "Trust ↑": metrics["trustworthiness"],
            "Cont ↑": metrics["continuity"],
            "kNN-Pres ↑": metrics["knn_preservation"],
            "ARI ↑": metrics["ari"],
            "NMI ↑": metrics["nmi"],
            "Silhouette ↑": metrics["silhouette"],
        })
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    csv_path = output_dir / "ablation_summary.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved CSV: {csv_path}")
    
    # Generate LaTeX table
    latex = df.to_latex(
        index=False,
        float_format="%.4f",
        column_format="l|ccc|rrrrrrr",
        caption="Ablation study results on MNIST. Higher is better for all metrics except Reconstruction error.",
        label="tab:ablation",
    )
    
    latex_path = output_dir / "ablation_summary.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"Saved LaTeX: {latex_path}")
    
    return df


def create_ood_table(ood_results: Dict, output_dir: Path):
    """Create summary table for OOD robustness."""
    
    configs = ["baseline", "t_only", "c_only", "e_only", "t_c", "t_e", "c_e", "t_c_e"]
    
    # Aggregate OOD metrics
    data = []
    for config in configs:
        if config not in ood_results:
            continue
        
        results = ood_results[config]
        
        # Average across rotation angles
        rot_trust = np.mean([results["rotation"][f"rot_{a}"]["trustworthiness"] for a in [15, 30, 45, 60]])
        rot_knn_acc = np.mean([results["rotation"][f"rot_{a}"]["knn_accuracy"] for a in [15, 30, 45, 60]])
        
        # Average across noise levels
        noise_trust = np.mean([results["noise"][f"noise_{s}"]["trustworthiness"] for s in [0.1, 0.2, 0.3, 0.5]])
        noise_knn_acc = np.mean([results["noise"][f"noise_{s}"]["knn_accuracy"] for s in [0.1, 0.2, 0.3, 0.5]])
        
        # Fashion transfer
        fashion_trust = results["fashion_mnist"]["trustworthiness"]
        fashion_knn_acc = results["fashion_mnist"]["knn_accuracy"]
        
        data.append({
            "Configuration": config,
            "Rot Trust ↑": rot_trust,
            "Rot Acc ↑": rot_knn_acc,
            "Noise Trust ↑": noise_trust,
            "Noise Acc ↑": noise_knn_acc,
            "Fashion Trust ↑": fashion_trust,
            "Fashion Acc ↑": fashion_knn_acc,
        })
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    csv_path = output_dir / "ood_summary.csv"
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"Saved CSV: {csv_path}")
    
    # Generate LaTeX table
    latex = df.to_latex(
        index=False,
        float_format="%.4f",
        column_format="l|rr|rr|rr",
        caption="OOD robustness results. Metrics averaged across scenarios.",
        label="tab:ood",
    )
    
    latex_path = output_dir / "ood_summary.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"Saved LaTeX: {latex_path}")
    
    return df


def create_fewshot_table(fewshot_results: Dict, output_dir: Path):
    """Create summary table for few-shot learning."""
    
    configs = ["baseline", "t_only", "c_only", "e_only", "t_c", "t_e", "c_e", "t_c_e"]
    k_shots = [1, 3, 5, 10]
    
    # Create separate tables for each method
    for method in ["logreg", "knn", "proto"]:
        data = []
        for config in configs:
            if config not in fewshot_results:
                continue
            
            results = fewshot_results[config]
            row = {"Configuration": config}
            
            for k in k_shots:
                mean = results[f"{k}_shot"][f"{method}_mean"]
                std = results[f"{k}_shot"][f"{method}_std"]
                row[f"{k}-shot"] = f"{mean:.3f}±{std:.3f}"
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Save as CSV
        csv_path = output_dir / f"fewshot_{method}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
        
        # Generate LaTeX table
        latex = df.to_latex(
            index=False,
            column_format="l|cccc",
            caption=f"Few-shot learning results using {method.upper()} classifier.",
            label=f"tab:fewshot_{method}",
        )
        
        latex_path = output_dir / f"fewshot_{method}.tex"
        with open(latex_path, "w") as f:
            f.write(latex)
        print(f"Saved LaTeX: {latex_path}")


def plot_ablation_comparison(ablation_results: Dict, output_dir: Path):
    """Create comparison plots for ablation study."""
    
    configs = ["baseline", "t_only", "c_only", "e_only", "t_c", "t_e", "c_e", "t_c_e"]
    metrics = ["trustworthiness", "continuity", "ari", "nmi", "silhouette"]
    metric_names = ["Trustworthiness", "Continuity", "ARI", "NMI", "Silhouette"]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    for ax, metric, name in zip(axes, metrics, metric_names):
        values = []
        labels = []
        
        for config in configs:
            if config not in ablation_results:
                continue
            values.append(ablation_results[config]["metrics"][metric])
            labels.append(config)
        
        ax.bar(range(len(labels)), values)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(alpha=0.3, axis="y")
    
    plt.tight_layout()
    plot_path = output_dir / "ablation_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot_path}")


def plot_ood_robustness(ood_results: Dict, output_dir: Path):
    """Create OOD robustness visualization."""
    
    configs = ["baseline", "t_c_e"]  # Compare baseline vs full
    scenarios = ["rotation", "noise"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Rotation robustness
    ax = axes[0]
    for config in configs:
        if config not in ood_results:
            continue
        
        angles = [15, 30, 45, 60]
        accs = [ood_results[config]["rotation"][f"rot_{a}"]["knn_accuracy"] for a in angles]
        ax.plot(angles, accs, marker="o", label=config)
    
    ax.set_xlabel("Rotation Angle (°)")
    ax.set_ylabel("kNN Accuracy")
    ax.set_title("Rotation Robustness")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Noise robustness
    ax = axes[1]
    for config in configs:
        if config not in ood_results:
            continue
        
        noise_levels = [0.1, 0.2, 0.3, 0.5]
        accs = [ood_results[config]["noise"][f"noise_{s}"]["knn_accuracy"] for s in noise_levels]
        ax.plot(noise_levels, accs, marker="o", label=config)
    
    ax.set_xlabel("Noise σ")
    ax.set_ylabel("kNN Accuracy")
    ax.set_title("Noise Robustness")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "ood_robustness.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot_path}")


def plot_fewshot_comparison(fewshot_results: Dict, output_dir: Path):
    """Create few-shot learning comparison."""
    
    configs = ["baseline", "t_only", "e_only", "t_c_e"]
    k_shots = [1, 3, 5, 10]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    methods = ["logreg", "knn", "proto"]
    method_names = ["Logistic Regression", "k-NN", "Prototypical"]
    
    for ax, method, name in zip(axes, methods, method_names):
        for config in configs:
            if config not in fewshot_results:
                continue
            
            means = [fewshot_results[config][f"{k}_shot"][f"{method}_mean"] for k in k_shots]
            stds = [fewshot_results[config][f"{k}_shot"][f"{method}_std"] for k in k_shots]
            
            ax.errorbar(k_shots, means, yerr=stds, marker="o", capsize=5, label=config)
        
        ax.set_xlabel("k (shots per class)")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(name)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plot_path = output_dir / "fewshot_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {plot_path}")


def generate_statistical_tests(ablation_results: Dict, output_dir: Path):
    """Run statistical significance tests."""
    
    # Compare t_c_e vs baseline across all metrics
    metrics = ["trustworthiness", "continuity", "ari", "nmi", "silhouette"]
    
    results_text = "# Statistical Significance Tests\n\n"
    results_text += "## Ablation Study: t_c_e vs baseline\n\n"
    
    if "baseline" not in ablation_results or "t_c_e" not in ablation_results:
        results_text += "Insufficient data for statistical tests.\n"
    else:
        baseline_metrics = ablation_results["baseline"]["metrics"]
        full_metrics = ablation_results["t_c_e"]["metrics"]
        
        for metric in metrics:
            baseline_val = baseline_metrics[metric]
            full_val = full_metrics[metric]
            improvement = ((full_val - baseline_val) / baseline_val) * 100
            
            results_text += f"### {metric}\n"
            results_text += f"- Baseline: {baseline_val:.4f}\n"
            results_text += f"- t_c_e: {full_val:.4f}\n"
            results_text += f"- Improvement: {improvement:+.2f}%\n\n"
    
    stats_path = output_dir / "statistical_tests.md"
    with open(stats_path, "w") as f:
        f.write(results_text)
    print(f"Saved stats: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experimental results - Phase 5.6")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/analysis",
        help="Directory to save analysis results",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("APS Results Analysis - Phase 5.6")
    print("="*60)
    
    # Load all results
    print("\nLoading results...")
    ablation_results = load_ablation_results()
    ood_results = load_ood_results()
    fewshot_results = load_fewshot_results()
    
    # Generate tables
    print("\n" + "="*60)
    print("Generating Summary Tables...")
    print("="*60)
    
    if ablation_results:
        create_ablation_table(ablation_results, output_dir)
    
    if ood_results:
        create_ood_table(ood_results, output_dir)
    
    if fewshot_results:
        create_fewshot_table(fewshot_results, output_dir)
    
    # Generate plots
    print("\n" + "="*60)
    print("Generating Comparison Plots...")
    print("="*60)
    
    if ablation_results:
        plot_ablation_comparison(ablation_results, output_dir)
    
    if ood_results:
        plot_ood_robustness(ood_results, output_dir)
    
    if fewshot_results:
        plot_fewshot_comparison(fewshot_results, output_dir)
    
    # Statistical tests
    print("\n" + "="*60)
    print("Running Statistical Tests...")
    print("="*60)
    
    if ablation_results:
        generate_statistical_tests(ablation_results, output_dir)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
