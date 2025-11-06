#!/bin/bash
# Quick test of fine-tuning pipeline with minimal data
# Run this first to verify everything works before full experiments

set -e

echo "========================================================================"
echo "Quick Fine-Tuning Test (1 epoch, 100 samples)"
echo "========================================================================"

cd "$(dirname "$0")/.."

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run quick baseline test
python scripts/run_phase006b_text_ood_finetune.py \
    --experiment baseline \
    --epochs 1 \
    --batch-size 16 \
    --max-samples 100 \
    --freeze-layers 10 \
    --latent-dim 16

echo ""
echo "✓ Quick test passed! Ready for full experiments."
echo "To run all experiments:"
echo "  bash scripts/run_finetune_experiments_all.sh"
