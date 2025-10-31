"""Analyze Phase 006B experiment results.

Compares OOD accuracy across different APS configurations and generates
summary tables and statistics.
"""

import json
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, List


def load_experiment_results(output_dir: str = './outputs/phase006b') -> Dict:
    """Load results from all experiments."""
    output_dir = Path(output_dir)
    
    experiments = ['baseline', 'aps-T', 'aps-C', 'aps-TC', 'aps-full']
    results = {}
    
    for exp in experiments:
        exp_dir = output_dir / exp
        
        # Load metrics
        metrics_file = exp_dir / 'final_metrics.json'
        history_file = exp_dir / 'history.json'
        config_file = exp_dir / 'config.json'
        
        if not metrics_file.exists():
            print(f"Warning: {exp} results not found, skipping...")
            continue
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        results[exp] = {
            'metrics': metrics,
            'history': history,
            'config': config
        }
    
    return results


def create_summary_table(results: Dict) -> pd.DataFrame:
    """Create summary comparison table."""
    rows = []
    
    for exp_name, data in results.items():
        metrics = data['metrics']
        config = data['config']
        history = data['history']
        
        # Final accuracies
        train_acc = history['train_acc'][-1] if history['train_acc'] else 0
        ood_acc = metrics['final_ood_accuracy']
        best_ood_acc = metrics['best_ood_accuracy']
        
        # OOD gap
        ood_gap = train_acc - ood_acc
        
        # Loss weights
        lambda_T = config['lambda_T']
        lambda_C = config['lambda_C']
        lambda_E = config['lambda_E']
        
        rows.append({
            'Experiment': exp_name,
            'λ_T': lambda_T,
            'λ_C': lambda_C,
            'λ_E': lambda_E,
            'Train Acc (%)': f"{100*train_acc:.2f}",
            'OOD Acc (%)': f"{100*ood_acc:.2f}",
            'Best OOD (%)': f"{100*best_ood_acc:.2f}",
            'OOD Gap (%)': f"{100*ood_gap:.2f}",
        })
    
    df = pd.DataFrame(rows)
    return df


def compute_improvements(results: Dict) -> Dict:
    """Compute improvements over baseline."""
    if 'baseline' not in results:
        return {}
    
    baseline_ood = results['baseline']['metrics']['final_ood_accuracy']
    
    improvements = {}
    for exp_name, data in results.items():
        if exp_name == 'baseline':
            improvements[exp_name] = {
                'absolute': 0.0,
                'relative': 0.0
            }
        else:
            ood_acc = data['metrics']['final_ood_accuracy']
            absolute_imp = ood_acc - baseline_ood
            relative_imp = (absolute_imp / baseline_ood) * 100 if baseline_ood > 0 else 0
            
            improvements[exp_name] = {
                'absolute': absolute_imp,
                'relative': relative_imp
            }
    
    return improvements


def print_summary(results: Dict):
    """Print comprehensive summary."""
    print("\n" + "="*80)
    print("Phase 006B: Experiment Results Summary")
    print("="*80 + "\n")
    
    # Summary table
    df = create_summary_table(results)
    print("Overall Results:")
    print("-" * 80)
    print(df.to_string(index=False))
    print()
    
    # Improvements over baseline
    improvements = compute_improvements(results)
    
    if improvements:
        print("\nImprovements over Baseline:")
        print("-" * 80)
        for exp_name, imp in improvements.items():
            if exp_name != 'baseline':
                ood_acc = results[exp_name]['metrics']['final_ood_accuracy']
                print(f"{exp_name:12s}: {100*ood_acc:.2f}% "
                      f"(+{100*imp['absolute']:.2f}%, {imp['relative']:+.1f}%)")
        print()
    
    # Best performing
    best_exp = max(results.items(), 
                   key=lambda x: x[1]['metrics']['final_ood_accuracy'])
    print(f"\nBest OOD Performance: {best_exp[0]}")
    print(f"  OOD Accuracy: {100*best_exp[1]['metrics']['final_ood_accuracy']:.2f}%")
    print()
    
    # Training stability
    print("\nTraining Stability (OOD accuracy std over last 10 epochs):")
    print("-" * 80)
    for exp_name, data in results.items():
        history = data['history']
        if len(history['ood_acc']) >= 10:
            last_10 = history['ood_acc'][-10:]
            std = np.std(last_10)
            print(f"{exp_name:12s}: {100*std:.2f}%")
    print()


def save_results_table(results: Dict, output_dir: str = './outputs/phase006b'):
    """Save summary table to CSV and Markdown."""
    output_dir = Path(output_dir)
    
    df = create_summary_table(results)
    
    # Save CSV
    csv_file = output_dir / 'results_summary.csv'
    df.to_csv(csv_file, index=False)
    print(f"Saved CSV: {csv_file}")
    
    # Save Markdown
    md_file = output_dir / 'results_summary_table.md'
    with open(md_file, 'w') as f:
        f.write("# Phase 006B Results Summary\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Improvements over Baseline\n\n")
        
        improvements = compute_improvements(results)
        for exp_name, imp in improvements.items():
            if exp_name != 'baseline':
                ood_acc = results[exp_name]['metrics']['final_ood_accuracy']
                f.write(f"- **{exp_name}**: {100*ood_acc:.2f}% "
                       f"(+{100*imp['absolute']:.2f}%, {imp['relative']:+.1f}%)\n")
    
    print(f"Saved Markdown: {md_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze Phase 006B results')
    parser.add_argument('--output-dir', type=str, default='./outputs/phase006b',
                       help='Output directory')
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.output_dir}...")
    results = load_experiment_results(args.output_dir)
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found {len(results)} experiments: {list(results.keys())}")
    
    # Print summary
    print_summary(results)
    
    # Save tables
    save_results_table(results, args.output_dir)
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
