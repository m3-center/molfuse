#!/bin/bash
# Quick MF cloud size verification
# Usage: bash scripts/count_mf_clouds.sh

echo "================================================================================"
echo "MF CLOUD SIZE VERIFICATION: Raw Line Counts"
echo "================================================================================"
echo ""

BASE_DIR="output_recalculated_full_datasets/datasets_2d_all"

# Phase 3 & 4 use the same Transferase datasets
echo "Phase 3 & 4: Transferase (KW-0808)"
echo "--------------------------------------------------------------------------------"
if [ -f "$BASE_DIR/KW-0808_Transferase_affinity_extracted_features.csv" ]; then
    lines=$(wc -l < "$BASE_DIR/KW-0808_Transferase_affinity_extracted_features.csv")
    compounds=$((lines - 1))  # Subtract header
    echo "  Features CSV:     $compounds rows (excluding header)"
else
    echo "  Features CSV:     NOT FOUND"
fi

if [ -f "$BASE_DIR/KW-0808_Transferase_affinity_extracted_fingerprints_ECFP4.csv" ]; then
    lines=$(wc -l < "$BASE_DIR/KW-0808_Transferase_affinity_extracted_fingerprints_ECFP4.csv")
    compounds=$((lines - 1))
    echo "  Fingerprints CSV: $compounds rows (excluding header)"
else
    echo "  Fingerprints CSV: NOT FOUND"
fi
echo ""

# Other Phase 4 targets
echo "Phase 4: Other Targets"
echo "--------------------------------------------------------------------------------"

targets=(
    "KW-0049_Antioxidant"
    "KW-0929_Antimicrobial"
    "KW-0505_Motor_protein"
    "KW-0202_Cytokine"
    "KW-0358_Heparin-binding"
    "KW-0456_Lyase"
    "KW-0560_Oxidoreductase"
)

for target in "${targets[@]}"; do
    feature_file="$BASE_DIR/${target}_affinity_extracted_features.csv"
    if [ -f "$feature_file" ]; then
        lines=$(wc -l < "$feature_file")
        compounds=$((lines - 1))
        printf "  %-30s %10d rows\n" "$target:" "$compounds"
    else
        printf "  %-30s NOT FOUND\n" "$target:"
    fi
done

echo ""
echo "================================================================================"
echo "Expected vs Actual (from summary CSVs)"
echo "================================================================================"
echo ""
echo "Phase 3 (UMAP/features, 'full'):"
echo "  Reported in phase3_summary_aggregated.csv: 96,657"
echo "  Actual CSV lines (Transferase features):   [see above]"
echo ""
echo "Phase 4 (UMAP/features, Transferase):"
echo "  Reported in phase4_summary_aggregated.csv: 187,034"
echo "  Actual CSV lines (Transferase features):   [see above]"
echo ""
echo "NOTE: Differences arise from:"
echo "  1. Deduplication by SMILES"
echo "  2. Affinity cutoff filtering (100 nM vs 100,000 nM)"
echo "  3. Target exclusion (removing actives from MF cloud)"
echo "  4. ZINC overlap removal"
echo ""
echo "Run the Python script for detailed analysis:"
echo "  python scripts/verify_mf_cloud_sizes.py"
echo "================================================================================"
