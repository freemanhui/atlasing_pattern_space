#!/bin/bash
# Repository cleanup script
# Removes unnecessary files and organizes the repo for public release

echo "🧹 Cleaning up repository..."

# Remove old README backup
echo "  - Removing README backup..."
rm -f README_old.md

# Remove standalone test files (tests are in tests/ directory)
echo "  - Removing standalone test files..."
rm -f test_energy_comparison.py
rm -f test_energy_comparison_3d.py
rm -f test_energy_detailed_3d.py

# Remove experimental/temporary files
echo "  - Removing experimental files..."
rm -f EXPERIMENTS_RUNNING.md
rm -f ROADMAP.md
rm -f ROADMAP_V2.md
rm -f CRITIQUE_RESPONSE.md

# Remove temporary doc files (keep essential docs)
echo "  - Cleaning up docs directory..."
cd docs
rm -f FINAL_SUMMARY.md
rm -f PAPER_TASKS_STATUS.md
rm -f STATUS_AND_RECOMMENDATIONS.md
rm -f TASK2_NLP_FINETUNING_COMPLETE.md
rm -f TASK4_TC_CONFLICT_COMPLETE.md
rm -f boundary_conditions_paper_outline.md
rm -f colored_mnist_results.md
rm -f colored_mnist_v3_results.md
rm -f "critical review APS.md"
rm -f day1_progress_summary.md
rm -f implementation_plan.md
rm -f nlp_finetuning_experiment.md
rm -f phase006_analysis_summary.md
rm -f phase006b_failure_analysis.md
rm -f phase1_final_summary.md
rm -f progress_phase1.md
rm -f tc_conflict_experiment_design.md
cd ..

# Remove paper build artifacts (keep final PDF and sources)
echo "  - Cleaning up paper directory..."
cd paper
rm -f MERGE_SUMMARY.md
rm -f PDF_COMPILATION_SUMMARY.md
rm -f README.md
rm -f REVIEWER_RESPONSE_SUMMARY.md
cd ..

# Remove .DS_Store files
echo "  - Removing .DS_Store files..."
find . -name ".DS_Store" -delete

# Remove Python cache
echo "  - Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# Clean up outputs (keep structure but remove temp files)
echo "  - Cleaning up outputs directory..."
find outputs -name "*.log" -delete 2>/dev/null
find outputs -name "*.tmp" -delete 2>/dev/null

echo "✅ Cleanup complete!"
echo ""
echo "📋 Summary:"
echo "  - Removed standalone test files"
echo "  - Removed experimental/temporary documentation"
echo "  - Removed build artifacts"
echo "  - Removed cache files"
echo "  - Removed .DS_Store files"
echo ""
echo "📁 Essential files retained:"
echo "  - README.md (updated)"
echo "  - WARP.md (development guide)"
echo "  - src/aps/ (core code)"
echo "  - scripts/ (experiment runners)"
echo "  - paper/ (research paper + figures)"
echo "  - outputs/ (experiment results)"
echo "  - tests/ (unit tests)"
echo ""
echo "🚀 Repository is ready for public release!"
