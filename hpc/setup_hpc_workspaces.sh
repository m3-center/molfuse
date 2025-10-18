#!/bin/bash
# =============================================================================
# UMMBAS v3.0 - Workspace Setup Script for HPC
# =============================================================================
# This script creates experimental workspace directories on the cluster's
# /work partition (high storage capacity) and creates symbolic links in the
# repository root for easy access.
#
# Usage:
#   bash setup_hpc_workspaces.sh
#
# Run this on the HPC cluster BEFORE submitting jobs.
# =============================================================================

# --- Configuration ---
WORK_BASE_DIR="/work/ahagg2s/ummbas_results"
REPO_ROOT="$(pwd)"

# Define all workspace directories for v3.0
WORKSPACES=(
    "experiment_workspace_v3_phase1"
    "experiment_workspace_v3_phase2"
    "experiment_workspace_v3_phase3"
    "experiment_workspace_v3_phase4"
)

# --- Header ---
echo "============================================================"
echo "UMMBAS v3.0 - HPC Workspace Setup"
echo "============================================================"
echo "Work directory: ${WORK_BASE_DIR}"
echo "Repository root: ${REPO_ROOT}"
echo "Workspaces to create: ${#WORKSPACES[@]}"
echo "============================================================"
echo ""

# --- Validate Paths ---
if [ ! -d "${REPO_ROOT}" ]; then
    echo "ERROR: Repository root not found: ${REPO_ROOT}"
    echo "Please run this script from the repository root."
    exit 1
fi

if [ ! -f "${REPO_ROOT}/main_orchestrator.py" ]; then
    echo "ERROR: main_orchestrator.py not found in ${REPO_ROOT}"
    echo "Please run this script from the repository root."
    exit 1
fi

# --- Create Base Work Directory ---
echo "Step 1: Creating base work directory..."
if [ ! -d "${WORK_BASE_DIR}" ]; then
    mkdir -p "${WORK_BASE_DIR}"
    if [ $? -eq 0 ]; then
        echo "✓ Created: ${WORK_BASE_DIR}"
    else
        echo "✗ Failed to create: ${WORK_BASE_DIR}"
        exit 1
    fi
else
    echo "✓ Already exists: ${WORK_BASE_DIR}"
fi
echo ""

# --- Create Workspace Directories and Symlinks ---
echo "Step 2: Creating workspace directories and symlinks..."
echo "------------------------------------------------------------"

SUCCESS_COUNT=0
SKIP_COUNT=0
ERROR_COUNT=0

for workspace in "${WORKSPACES[@]}"; do
    WORK_PATH="${WORK_BASE_DIR}/${workspace}"
    LINK_PATH="${REPO_ROOT}/${workspace}"
    
    echo "Processing: ${workspace}"
    
    # Create directory in /work
    if [ ! -d "${WORK_PATH}" ]; then
        mkdir -p "${WORK_PATH}"
        if [ $? -eq 0 ]; then
            echo "  ✓ Created directory: ${WORK_PATH}"
        else
            echo "  ✗ Failed to create directory: ${WORK_PATH}"
            ERROR_COUNT=$((ERROR_COUNT + 1))
            continue
        fi
    else
        echo "  ✓ Directory exists: ${WORK_PATH}"
    fi
    
    # Create symlink in repository root
    if [ -L "${LINK_PATH}" ]; then
        # Symlink exists, check if it points to the right place
        CURRENT_TARGET=$(readlink -f "${LINK_PATH}")
        EXPECTED_TARGET=$(readlink -f "${WORK_PATH}")
        
        if [ "${CURRENT_TARGET}" == "${EXPECTED_TARGET}" ]; then
            echo "  ✓ Symlink already correct: ${workspace}"
            SKIP_COUNT=$((SKIP_COUNT + 1))
        else
            echo "  ⚠ Symlink exists but points elsewhere:"
            echo "    Current: ${CURRENT_TARGET}"
            echo "    Expected: ${EXPECTED_TARGET}"
            echo "    Removing old symlink and creating new one..."
            rm "${LINK_PATH}"
            ln -s "${WORK_PATH}" "${LINK_PATH}"
            if [ $? -eq 0 ]; then
                echo "  ✓ Updated symlink: ${workspace}"
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
            else
                echo "  ✗ Failed to update symlink: ${workspace}"
                ERROR_COUNT=$((ERROR_COUNT + 1))
            fi
        fi
    elif [ -e "${LINK_PATH}" ]; then
        # Path exists but is not a symlink (regular dir or file)
        echo "  ⚠ Path exists as regular directory/file: ${LINK_PATH}"
        echo "    Please manually remove or rename it before running this script."
        ERROR_COUNT=$((ERROR_COUNT + 1))
    else
        # Create new symlink
        ln -s "${WORK_PATH}" "${LINK_PATH}"
        if [ $? -eq 0 ]; then
            echo "  ✓ Created symlink: ${workspace} -> ${WORK_PATH}"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "  ✗ Failed to create symlink: ${workspace}"
            ERROR_COUNT=$((ERROR_COUNT + 1))
        fi
    fi
    
    echo ""
done

# --- Summary ---
echo "============================================================"
echo "Setup Complete"
echo "============================================================"
echo "Symlinks created: ${SUCCESS_COUNT}"
echo "Already existed: ${SKIP_COUNT}"
echo "Errors: ${ERROR_COUNT}"
echo "============================================================"
echo ""

if [ ${ERROR_COUNT} -gt 0 ]; then
    echo "⚠ WARNING: Some workspaces failed to set up."
    echo "   Please review the errors above before proceeding."
    exit 1
fi

# --- Verify Setup ---
echo "Step 3: Verifying symlinks..."
echo "------------------------------------------------------------"

ALL_OK=true
for workspace in "${WORKSPACES[@]}"; do
    LINK_PATH="${REPO_ROOT}/${workspace}"
    
    if [ -L "${LINK_PATH}" ]; then
        TARGET=$(readlink "${LINK_PATH}")
        echo "✓ ${workspace} -> ${TARGET}"
    else
        echo "✗ ${workspace} (symlink missing or broken)"
        ALL_OK=false
    fi
done

echo "============================================================"

if [ "${ALL_OK}" = true ]; then
    echo "✓ All workspaces set up successfully!"
    echo ""
    echo "You can now run experiments. Results will be stored in:"
    echo "  ${WORK_BASE_DIR}/"
    echo ""
    echo "Next steps:"
    echo "  1. Generate configs: python generate_phase1_configs.py"
    echo "  2. Submit jobs: bash hpc/submit_v3_phase1.sh"
else
    echo "✗ Some workspaces have issues. Please fix before proceeding."
    exit 1
fi

echo "============================================================"
