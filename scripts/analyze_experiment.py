#!/usr/bin/env python3
"""
Analyze and visualize APS experiment results.

Usage:
    # Analyze single experiment
    python scripts/analyze_experiment.py --experiment aps_baseline
    
    # Compare multiple experiments
    python scripts/analyze_experiment.py --compare aps_baseline aps_T-only aps_full
    
    # Create plots for specific experiment
    python scripts/analyze_experiment.py --experiment aps_full --plots
"""

import argparse
from pathlib import Path
from aps.training.visualize import (
    plot_loss_curves,
    plot_step_metrics,
    plot_ablation_comparison,
    plot_all_components,
    create_training_summary,
)


def main():
    parser = argparse.ArgumentParser(description='Analyze APS experiment results')
    
    # Experiment selection
    parser.add_argument('--experiment', type=str,
                       help='Single experiment name to analyze')
    parser.add_argument('--compare', type=str, nargs='+',
                       help='Multiple experiment names to compare')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Base output directory')
    parser.add_argument('--plots', action='store_true',
                       help='Generate all plots')
    parser.add_argument('--summary', action='store_true',
                       help='Print training summary')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save plots to disk')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Single experiment analysis
    if args.experiment:
        experiment_dir = output_dir / args.experiment
        log_dir = experiment_dir / 'logs'
        plot_dir = experiment_dir / 'plots'
        
        if not log_dir.exists():
            print(f"Error: Experiment '{args.experiment}' not found at {experiment_dir}")
            return
        
        print(f"\nAnalyzing experiment: {args.experiment}")
        print("="*60)
        
        # Print summary
        if args.summary or not args.plots:
            summary = create_training_summary(log_dir)
            print(summary)
        
        # Generate plots
        if args.plots:
            if args.save_plots:
                plot_all_components(log_dir, save_dir=plot_dir)
                print(f"\nPlots saved to: {plot_dir}")
            else:
                # Just show loss curves
                plot_loss_curves(log_dir)
    
    # Multiple experiment comparison
    elif args.compare:
        print(f"\nComparing {len(args.compare)} experiments:")
        print("="*60)
        
        experiment_dirs = {}
        for exp_name in args.compare:
            exp_dir = output_dir / exp_name
            log_dir = exp_dir / 'logs'
            
            if not log_dir.exists():
                print(f"Warning: Skipping '{exp_name}' (not found)")
                continue
            
            experiment_dirs[exp_name] = log_dir
            
            # Print summary for each
            if args.summary:
                print(f"\n{exp_name}:")
                print("-"*40)
                summary = create_training_summary(log_dir)
                print(summary)
        
        if len(experiment_dirs) > 1:
            print(f"\nGenerating comparison plot...")
            
            if args.save_plots:
                save_path = output_dir / 'ablation_comparison.png'
                plot_ablation_comparison(experiment_dirs, save_path=save_path)
            else:
                plot_ablation_comparison(experiment_dirs)
        else:
            print("Need at least 2 valid experiments to compare")
    
    else:
        # List available experiments
        print("Available experiments:")
        print("="*60)
        
        if output_dir.exists():
            experiments = [d.name for d in output_dir.iterdir() if d.is_dir() and (d / 'logs').exists()]
            
            if experiments:
                for exp in sorted(experiments):
                    exp_dir = output_dir / exp / 'logs'
                    # Try to get epoch count
                    epoch_file = exp_dir / 'epoch_metrics.json'
                    if epoch_file.exists():
                        import json
                        with open(epoch_file) as f:
                            data = json.load(f)
                            n_epochs = len(data['epochs'])
                        print(f"  - {exp:30s} ({n_epochs} epochs)")
                    else:
                        print(f"  - {exp}")
                
                print(f"\nUse --experiment <name> to analyze a specific experiment")
                print(f"Use --compare <name1> <name2> ... to compare multiple experiments")
            else:
                print("No experiments found")
        else:
            print(f"Output directory not found: {output_dir}")


if __name__ == '__main__':
    main()
