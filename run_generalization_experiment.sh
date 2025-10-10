#!/bin/bash

# ============================================================================
# Quick Start Script for Generalization Experiment
# ============================================================================
# This script provides a convenient way to set up and run the generalization
# experiment step by step.
#
# Usage: bash run_generalization_experiment.sh [step]
#   step 1: Generate configurations
#   step 2: Submit jobs to HPC
#   step 3: Aggregate and analyze results
#   step all: Run all steps (use with caution - step 2 needs time to complete)
# ============================================================================

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo "========================================================================"
    echo "$1"
    echo "========================================================================"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Step 1: Generate configurations
step_generate_configs() {
    print_header "STEP 1: Generating Configuration Files"
    
    if [ -d "generalization_configs" ] && [ "$(ls -A generalization_configs 2>/dev/null)" ]; then
        print_warning "Configuration directory already exists with files."
        read -p "Regenerate configurations? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_warning "Skipping configuration generation."
            return 0
        fi
        rm -rf generalization_configs/*.json
    fi
    
    python generate_generalization_configs.py
    
    if [ $? -eq 0 ]; then
        CONFIG_COUNT=$(find generalization_configs -name "*.json" | wc -l)
        print_success "Generated ${CONFIG_COUNT} configuration files"
        echo ""
        echo "Configurations saved to: generalization_configs/"
        ls -1 generalization_configs/*.json | head -5
        if [ ${CONFIG_COUNT} -gt 5 ]; then
            echo "... and $((CONFIG_COUNT - 5)) more files"
        fi
    else
        print_error "Failed to generate configurations"
        exit 1
    fi
}

# Step 2: Submit jobs
step_submit_jobs() {
    print_header "STEP 2: Submitting Jobs to HPC"
    
    if [ ! -d "generalization_configs" ]; then
        print_error "Configuration directory not found. Run step 1 first."
        exit 1
    fi
    
    CONFIG_COUNT=$(find generalization_configs -name "*.json" | wc -l)
    if [ ${CONFIG_COUNT} -eq 0 ]; then
        print_error "No configuration files found. Run step 1 first."
        exit 1
    fi
    
    echo "About to submit ${CONFIG_COUNT} configs × 5 seeds = $((CONFIG_COUNT * 5)) jobs"
    echo ""
    read -p "Continue with job submission? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_warning "Job submission cancelled."
        return 0
    fi
    
    bash hpc/submit_generalization_jobs.sh
    
    if [ $? -eq 0 ]; then
        print_success "Jobs submitted successfully"
        echo ""
        echo "Monitor jobs with: squeue -u \$USER"
        echo "Check logs in: slurm_logs/"
        echo ""
        print_warning "Jobs need to complete before running step 3"
    else
        print_error "Job submission failed"
        exit 1
    fi
}

# Step 3: Aggregate results
step_aggregate_results() {
    print_header "STEP 3: Aggregating and Analyzing Results"
    
    if [ ! -d "experiment_workspace_generalization" ]; then
        print_error "Generalization workspace not found. Have the jobs completed?"
        exit 1
    fi
    
    # Check if there are any results
    RESULT_COUNT=$(find experiment_workspace_generalization -name "*_ranking_metrics.csv" 2>/dev/null | wc -l)
    if [ ${RESULT_COUNT} -eq 0 ]; then
        print_error "No results found in workspace. Have the jobs completed?"
        exit 1
    fi
    
    print_success "Found ${RESULT_COUNT} result files"
    echo ""
    
    python aggregate_generalization_analysis.py \
        --abl1_workspace experiment_workspace_rerun_hyperparam_sweep/ \
        --generalization_workspace experiment_workspace_generalization/ \
        --output_dir final_report_generalization/
    
    if [ $? -eq 0 ]; then
        print_success "Analysis complete"
        echo ""
        LATEST_REPORT=$(ls -td final_report_generalization/generalization_report_* 2>/dev/null | head -1)
        if [ -n "${LATEST_REPORT}" ]; then
            echo "Report saved to: ${LATEST_REPORT}"
            echo ""
            echo "Key files:"
            echo "  - Figures: ${LATEST_REPORT}/figures/"
            echo "  - Tables: ${LATEST_REPORT}/tables/"
            echo "  - Master data: ${LATEST_REPORT}/tables/master_generalization_metrics.csv"
        fi
    else
        print_error "Analysis failed"
        exit 1
    fi
}

# Show usage
show_usage() {
    cat << EOF
Usage: bash run_generalization_experiment.sh [STEP]

Steps:
  1, generate    Generate configuration files
  2, submit      Submit jobs to HPC cluster
  3, analyze     Aggregate and analyze results
  all            Run all steps (WARNING: step 2 needs time to complete)
  help           Show this help message

Examples:
  bash run_generalization_experiment.sh 1           # Generate configs only
  bash run_generalization_experiment.sh submit      # Submit jobs
  bash run_generalization_experiment.sh analyze     # Analyze results

For more information, see: GENERALIZATION_EXPERIMENT_README.md
EOF
}

# Main execution
main() {
    if [ $# -eq 0 ]; then
        show_usage
        exit 0
    fi
    
    case "$1" in
        1|generate)
            step_generate_configs
            ;;
        2|submit)
            step_submit_jobs
            ;;
        3|analyze)
            step_aggregate_results
            ;;
        all)
            print_warning "Running all steps. Note: Step 2 requires jobs to complete before step 3."
            read -p "Continue? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                step_generate_configs
                step_submit_jobs
                print_warning "Jobs submitted. Wait for them to complete, then run step 3 manually."
            fi
            ;;
        help|-h|--help)
            show_usage
            ;;
        *)
            print_error "Unknown step: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
