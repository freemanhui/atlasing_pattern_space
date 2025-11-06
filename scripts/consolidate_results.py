"""
Consolidate experimental results from various output directories
into a unified format for paper figure generation.
"""

import json
from pathlib import Path

# Define source directories and their mapping to paper versions
MAPPINGS = [
    {
        "version": "v2",  # 99.5% correlation (harder)
        "source_dir": "outputs/colored_mnist",
        "models": {
            "baseline": "baseline",
            "APS-T": "aps-t",
            "APS-C": "aps-c",
            "APS-Full": "aps-full"
        }
    },
    {
        "version": "v3.1",  # 99%/-99% anti-correlation
        "source_dir": "outputs/colored_mnist_v3",
        "models": {
            "baseline": "baseline",
            "APS-C": "aps-c",
            "APS-Full": "aps-full"
        }
    }
]

def consolidate_results():
    """Consolidate all results into summary format."""
    consolidated = {}
    
    for mapping in MAPPINGS:
        version = mapping["version"]
        source_dir = Path(mapping["source_dir"])
        
        if not source_dir.exists():
            print(f"Warning: {source_dir} does not exist, skipping {version}")
            continue
        
        version_results = {}
        
        for model_name, model_dir in mapping["models"].items():
            results_file = source_dir / model_dir / "results.json"
            
            if not results_file.exists():
                print(f"Warning: {results_file} does not exist, skipping {model_name} for {version}")
                continue
            
            with open(results_file) as f:
                data = json.load(f)
            
            # Extract key metrics
            version_results[model_name] = {
                "test_accuracy": data.get("best_test_accuracy", 0),
                "causal_ratio": data.get("causal_metrics", {}).get("causal_ratio", 0),
                "reliance_gap": data.get("causal_metrics", {}).get("reliance_gap", 0),
                "env_invariance": data.get("causal_metrics", {}).get("invariance_score", 0),
                "final_test_accuracy": data.get("final_test_accuracy", 0)
            }
        
        consolidated[version] = version_results
    
    # Save consolidated results
    output_dir = Path("outputs")
    summary_file = output_dir / "consolidated_summary.json"
    
    with open(summary_file, 'w') as f:
        json.dump(consolidated, f, indent=2)
    
    print(f"✓ Consolidated results saved to: {summary_file}")
    
    # Print summary table
    print("\nConsolidated Results Summary:")
    print("=" * 80)
    for version, models in consolidated.items():
        print(f"\n{version}:")
        for model, metrics in models.items():
            print(f"  {model:15s}: Acc={metrics['test_accuracy']:.4f}, "
                  f"Causal Ratio={metrics['causal_ratio']:.2f}")

if __name__ == "__main__":
    consolidate_results()
