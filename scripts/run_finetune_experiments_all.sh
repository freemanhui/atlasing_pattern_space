#!/bin/bash
# Master script to run all AG News fine-tuning experiments
# Addresses reviewer feedback: "Rerun AG News with BERT fine-tuning enabled"

set -e  # Exit on error

echo "========================================================================"
echo "AG News Fine-Tuning Experiments - All Configurations"
echo "========================================================================"
echo ""
echo "This will run 5 experiments with BERT fine-tuning enabled:"
echo "  1. Baseline (no APS components)"
echo "  2. APS-T (Topology only)"
echo "  3. APS-C (Causality only)"
echo "  4. APS-TC (Topology + Causality)"
echo "  5. APS-Full (T + C + E)"
echo ""
echo "Training settings:"
echo "  - BERT fine-tuning: 6 layers frozen, 6 layers trainable"
echo "  - Epochs: 10 (sufficient for fine-tuning)"
echo "  - Batch size: 32"
echo "  - Max samples: 5000 per domain"
echo "  - Estimated time: ~2-3 hours total (M1/M2 Mac or GPU)"
echo ""
echo "========================================================================"

# Change to project root
cd "$(dirname "$0")/.."

# Activate virtual environment if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Output directory
OUTPUT_DIR="./outputs/phase006b_finetune"
mkdir -p "$OUTPUT_DIR"

# Run all experiments
echo ""
echo "Starting experiments..."
echo ""

for exp in baseline aps-T aps-C aps-TC aps-full; do
    echo "------------------------------------------------------------------------"
    echo "Running: $exp"
    echo "------------------------------------------------------------------------"
    
    python scripts/run_phase006b_text_ood_finetune.py \
        --experiment "$exp" \
        --epochs 10 \
        --batch-size 32 \
        --max-samples 5000 \
        --freeze-layers 6 \
        --latent-dim 32
    
    echo ""
    echo "✓ Completed: $exp"
    echo ""
done

# Compare results
echo "========================================================================"
echo "All experiments completed!"
echo "========================================================================"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To analyze results, run:"
echo "  python scripts/analyze_finetune_results.py"
echo ""
