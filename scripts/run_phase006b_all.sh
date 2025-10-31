#!/bin/bash
# Run all Phase 006B experiments sequentially
# This will take several hours on CPU (faster with GPU)

set -e  # Exit on error

echo "=================================="
echo "Phase 006B: Full Experiment Suite"
echo "=================================="
echo ""
echo "This will run 5 experiments:"
echo "  1. baseline   (λ_T=0, λ_C=0, λ_E=0)"
echo "  2. aps-T      (λ_T=1.0)"
echo "  3. aps-C      (λ_C=1.0)"
echo "  4. aps-TC     (λ_T=1.0, λ_C=0.5)"
echo "  5. aps-full   (λ_T=1.0, λ_C=0.5, λ_E=0.1)"
echo ""
echo "Each experiment: 30 epochs, full dataset"
echo "Estimated time: ~2-3 hours per experiment"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    exit 1
fi

# Configuration
EPOCHS=30
BATCH_SIZE=128  # Optimized for M3 with 36GB memory
OUTPUT_DIR="./outputs/phase006b"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/experiments.log"
echo "Starting experiments at $(date)" > "$LOG_FILE"

# Function to run experiment
run_experiment() {
    local name=$1
    echo ""
    echo "=================================="
    echo "Running: $name"
    echo "=================================="
    echo "Starting $name at $(date)" >> "$LOG_FILE"
    
    python scripts/run_phase006b_text_ood.py \
        --experiment "$name" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee -a "$LOG_FILE"
    
    echo "Completed $name at $(date)" >> "$LOG_FILE"
    echo ""
}

# Run all experiments
run_experiment "baseline"
run_experiment "aps-T"
run_experiment "aps-C"
run_experiment "aps-TC"
run_experiment "aps-full"

echo ""
echo "=================================="
echo "All experiments completed!"
echo "=================================="
echo "Results saved to: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Next steps:"
echo "  1. python scripts/analyze_phase006b_results.py"
echo "  2. python scripts/visualize_phase006b_results.py"
