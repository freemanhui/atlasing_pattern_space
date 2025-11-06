#!/bin/bash
# Run all NLP fine-tuning experiments for Phase 006B
# This addresses reviewer feedback about frozen embeddings limiting APS effectiveness

set -e  # Exit on error

echo "========================================"
echo "NLP Fine-Tuning Experiments - Phase 006B"
echo "========================================"
echo ""
echo "This experiment tests if BERT fine-tuning (trainable embeddings)"
echo "allows T and C components to show benefits vs frozen embeddings."
echo ""
echo "Expected outcomes:"
echo "- Frozen (previous): T/C no benefit, E marginal (+0.11pp)"
echo "- Fine-tuned (this): T/C should improve OOD accuracy by 2-5pp"
echo ""

# Configuration
EPOCHS=10
MAX_SAMPLES=5000  # Smaller dataset for faster training
FREEZE_LAYERS=6    # Freeze first 6 BERT layers (fine-tune last 6)
BATCH_SIZE=32

echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Max samples per domain: $MAX_SAMPLES"
echo "  Freeze first $FREEZE_LAYERS BERT layers"
echo "  Batch size: $BATCH_SIZE"
echo ""

# Experiment 1: Baseline
echo "[1/5] Running baseline (no APS)..."
python scripts/run_phase006b_text_ood_finetune.py \
    --experiment baseline \
    --epochs $EPOCHS \
    --max-samples $MAX_SAMPLES \
    --freeze-layers $FREEZE_LAYERS \
    --batch-size $BATCH_SIZE

# Experiment 2: APS-T (Topology only)
echo ""
echo "[2/5] Running APS-T (topology only)..."
python scripts/run_phase006b_text_ood_finetune.py \
    --experiment aps-T \
    --epochs $EPOCHS \
    --max-samples $MAX_SAMPLES \
    --freeze-layers $FREEZE_LAYERS \
    --batch-size $BATCH_SIZE

# Experiment 3: APS-C (Causality only)
echo ""
echo "[3/5] Running APS-C (causality only)..."
python scripts/run_phase006b_text_ood_finetune.py \
    --experiment aps-C \
    --epochs $EPOCHS \
    --max-samples $MAX_SAMPLES \
    --freeze-layers $FREEZE_LAYERS \
    --batch-size $BATCH_SIZE

# Experiment 4: APS-TC (Topology + Causality)
echo ""
echo "[4/5] Running APS-TC (topology + causality)..."
python scripts/run_phase006b_text_ood_finetune.py \
    --experiment aps-TC \
    --epochs $EPOCHS \
    --max-samples $MAX_SAMPLES \
    --freeze-layers $FREEZE_LAYERS \
    --batch-size $BATCH_SIZE

# Experiment 5: APS-Full (T + C + E)
echo ""
echo "[5/5] Running APS-Full (topology + causality + energy)..."
python scripts/run_phase006b_text_ood_finetune.py \
    --experiment aps-full \
    --epochs $EPOCHS \
    --max-samples $MAX_SAMPLES \
    --freeze-layers $FREEZE_LAYERS \
    --batch-size $BATCH_SIZE

echo ""
echo "========================================"
echo "All experiments complete!"
echo "========================================"
echo ""
echo "Results saved to: outputs/phase006b_finetune/"
echo ""
echo "To analyze results, run:"
echo "  python scripts/analyze_finetune_results.py"
echo ""
