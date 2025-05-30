TEST_PREPARE_OUT_DIR="test_outputs/prepare_data_tests/target1_output"
TEST_CONFIG_PATH="debugging_config.json" # Assuming test_config.json is set up for "PyruvateKinaseM2_P14618" as the first target
TARGET_ID_NAME_TEST="PyruvateKinaseM2_P14618" # From your test_config.json
RDKIT_FEATURES_JSON_STRING='["DipoleMoment","ABC","nAcid","nBase","nAromAtom","nAtom","nH","nC","nN","nO","nS","nP","nX","nBonds","nBondsO","nBondsS","nBondsD","nBondsT","nBondsA","nBondsM","nBondsKS","nBondsKD","EState_VSA7","nHBAcc","nHBDon","Lipinski","apol","bpol","nRing","n3Ring","n4Ring","n5Ring","n6Ring","n7Ring","n8Ring","nRot","Diameter","TopoShapeIndex","Vabc","MW"]'
RAW_TARGET_LIGANDS_CSV="$TEST_PREPARE_OUT_DIR/${TARGET_ID_NAME_TEST}_target_ligands_for_feature_calc_raw.csv"

# Assume 'features' representation for this example
REPR_TYPE_TEST="features"
TARGET_LIGANDS_UNSCALED_INPUT="$TEST_MOLCALCS_OUT_DIR/${TARGET_ID_NAME_TEST}_target_ligands_project_${REPR_TYPE_TEST}.csv" # or from cache_run1 etc.


TEST_MOLCALCS_OUT_DIR="test_outputs/molcalcs_tests"

echo "Running feature and fingerprint calculations for target: $TARGET_ID_NAME_TEST"
python core_scripts/calculate_features_and_fingerprints_exp.py \
    --input_csv "$RAW_TARGET_LIGANDS_CSV" \
    --output_dir "$TEST_MOLCALCS_OUT_DIR" \
    --representation_type "features" \
    --file_label "${TARGET_ID_NAME_TEST}_target_ligands_project" \
    --n_jobs 1 \

echo "Running fingerprint calculations for target: $TARGET_ID_NAME_TEST"
python core_scripts/calculate_features_and_fingerprints_exp.py \
    --input_csv "$RAW_TARGET_LIGANDS_CSV" \
    --output_dir "$TEST_MOLCALCS_OUT_DIR" \
    --representation_type "fingerprints" \
    --file_label "${TARGET_ID_NAME_TEST}_target_ligands_project" \
    --n_jobs 1 \

TEST_SIMSPACE_OUT_DIR="test_outputs/simspace_tests/simspaces_out"
TEST_SIMSPACE_MODELS_DIR="test_outputs/simspace_tests/models_out"
mkdir -p "$TEST_SIMSPACE_OUT_DIR" "$TEST_SIMSPACE_MODELS_DIR"
SIMSPACE_DIM_TEST=2
TARGET_LIGANDS_UNSCALED_INPUT="$TEST_MOLCALCS_OUT_DIR/${TARGET_ID_NAME_TEST}_target_ligands_project_${REPR_TYPE_TEST}.csv" # or from cache_run1 etc.
CHEMBL_MF_EXCLUDED_INPUT="$TEST_PREPARE_OUT_DIR/${TARGET_ID_NAME_TEST}_chembl_mf_excluded_${REPR_TYPE_TEST}.csv"
ZINC_EXCLUDED_INPUT="$TEST_PREPARE_OUT_DIR/${TARGET_ID_NAME_TEST}_zinc_excluded_${REPR_TYPE_TEST}.csv"

echo "Running similarity space calculations for target: $TARGET_ID_NAME_TEST with DR method PCA"
python core_scripts/calculate_similarityspaces_exp.py \
    --chembl_mf_data_path "$CHEMBL_MF_EXCLUDED_INPUT" \
    --zinc_data_path "$ZINC_EXCLUDED_INPUT" \
    --target_ligands_unscaled_path_for_tsne "$TARGET_LIGANDS_UNSCALED_INPUT" \
    --simspace_dim $SIMSPACE_DIM_TEST \
    --representation_type "$REPR_TYPE_TEST" \
    --target_id_name "$TARGET_ID_NAME_TEST" \
    --output_simspace_dir "$TEST_SIMSPACE_OUT_DIR" \
    --output_model_dir "$TEST_SIMSPACE_MODELS_DIR" \
    --rdkit_features_list_target_str "$RDKIT_FEATURES_JSON_STRING" \
    --dr_method_pca True

echo "Running similarity space calculations for target: $TARGET_ID_NAME_TEST with DR method UMAP"
python core_scripts/calculate_similarityspaces_exp.py \
    --chembl_mf_data_path "$CHEMBL_MF_EXCLUDED_INPUT" \
    --zinc_data_path "$ZINC_EXCLUDED_INPUT" \
    --target_ligands_unscaled_path_for_tsne "$TARGET_LIGANDS_UNSCALED_INPUT" \
    --simspace_dim $SIMSPACE_DIM_TEST \
    --representation_type "$REPR_TYPE_TEST" \
    --target_id_name "$TARGET_ID_NAME_TEST" \
    --output_simspace_dir "$TEST_SIMSPACE_OUT_DIR" \
    --output_model_dir "$TEST_SIMSPACE_MODELS_DIR" \
    --rdkit_features_list_target_str "$RDKIT_FEATURES_JSON_STRING" \
    --dr_method_umap True \
    --umap_metric_to_run_euclidean

echo "Running similarity space calculations for target: $TARGET_ID_NAME_TEST with DR method t-SNE"
echo "TARGET_LIGANDS_UNSCALED_INPUT: $TARGET_LIGANDS_UNSCALED_INPUT"
python core_scripts/calculate_similarityspaces_exp.py \
    --chembl_mf_data_path "$CHEMBL_MF_EXCLUDED_INPUT" \
    --zinc_data_path "$ZINC_EXCLUDED_INPUT" \
    --target_ligands_unscaled_path_for_tsne "$TARGET_LIGANDS_UNSCALED_INPUT" \
    --simspace_dim $SIMSPACE_DIM_TEST \
    --representation_type "$REPR_TYPE_TEST" \
    --target_id_name "$TARGET_ID_NAME_TEST" \
    --output_simspace_dir "$TEST_SIMSPACE_OUT_DIR" \
    --output_model_dir "$TEST_SIMSPACE_MODELS_DIR" \
    --rdkit_features_list_target_str "$RDKIT_FEATURES_JSON_STRING" \
    --dr_method_tsne True \
    --tsne_perplexity 5 \
    --tsne_pca_components 10 # Smaller values for faster test

mkdir -p "$TEST_SIMSPACE_OUT_DIR/all_dr" "$TEST_SIMSPACE_MODELS_DIR/all_dr"
echo "Running similarity space calculations for target: $TARGET_ID_NAME_TEST with all DR methods"
python core_scripts/calculate_similarityspaces_exp.py \
    --chembl_mf_data_path "$CHEMBL_MF_EXCLUDED_INPUT" \
    --zinc_data_path "$ZINC_EXCLUDED_INPUT" \
    --target_ligands_unscaled_path_for_tsne "$TARGET_LIGANDS_UNSCALED_INPUT" \
    --simspace_dim $SIMSPACE_DIM_TEST \
    --representation_type "$REPR_TYPE_TEST" \
    --target_id_name "$TARGET_ID_NAME_TEST" \
    --output_simspace_dir "$TEST_SIMSPACE_OUT_DIR/all_dr" \
    --output_model_dir "$TEST_SIMSPACE_MODELS_DIR/all_dr" \
    --rdkit_features_list_target_str "$RDKIT_FEATURES_JSON_STRING" \
    --dr_method_pca True \
    --dr_method_umap True \
    --umap_metric_to_run_euclidean \
    --umap_metric_to_run_cosine \
    --dr_method_tsne True \
    --tsne_perplexity 5 \
    --tsne_pca_components 10

# Using outputs from "all_dr" test of calculate_similarityspaces_exp.py
COMPREHENSIVE_SIMSPACE_CSV="$TEST_SIMSPACE_OUT_DIR/all_dr/${TARGET_ID_NAME_TEST}_${REPR_TYPE_TEST}_dim${SIMSPACE_DIM_TEST}_similarity_space.csv"
MODELS_INPUT_DIR="$TEST_SIMSPACE_MODELS_DIR/all_dr" # Directory containing scaler, PCA, UMAP models

# Featurized target ligands
PROCESSED_TARGET_LIGANDS_INPUT="$TEST_MOLCALCS_OUT_DIR/${TARGET_ID_NAME_TEST}_target_ligands_project_${REPR_TYPE_TEST}.csv" 

TEST_PROJECT_ANALYZE_OUT_DIR_BASE="test_outputs/project_analyze_tests"
K_FOR_KNN_TEST="3,5"
MODEL_NAME_ROOT_PROJ="${TARGET_ID_NAME_TEST}_${REPR_TYPE_TEST}_dim${SIMSPACE_DIM_TEST}"

DR_KEY_TEST="pca"
DR_SHORT_NAME_TEST="PCA"
TEST_PROJECT_ANALYZE_OUT_DIR="$TEST_PROJECT_ANALYZE_OUT_DIR_BASE/$DR_SHORT_NAME_TEST"
mkdir -p "$TEST_PROJECT_ANALYZE_OUT_DIR"
python experimental_pipeline/project_and_analyze.py \
    --target_ligands_repr_path "$PROCESSED_TARGET_LIGANDS_INPUT" \
    --simspace_csv_path "$COMPREHENSIVE_SIMSPACE_CSV" \
    --model_dir_for_projection "$MODELS_INPUT_DIR" \
    --model_name_root_for_projection "$MODEL_NAME_ROOT_PROJ" \
    --dr_method_key "$DR_KEY_TEST" \
    --dr_short_name "$DR_SHORT_NAME_TEST" \
    --simspace_dim $SIMSPACE_DIM_TEST \
    --k_for_knn "$K_FOR_KNN_TEST" \
    --output_dir "$TEST_PROJECT_ANALYZE_OUT_DIR" \
    --target_id_name "$TARGET_ID_NAME_TEST" \
    --representation_type "$REPR_TYPE_TEST" \
    --rdkit_features_list_target_str "$RDKIT_FEATURES_JSON_STRING"

DR_KEY_TEST="umap_euclidean" # Must match a key in config.dimensionality_reduction_methods
DR_SHORT_NAME_TEST="UMAP-Euclidean"
TEST_PROJECT_ANALYZE_OUT_DIR="$TEST_PROJECT_ANALYZE_OUT_DIR_BASE/${DR_KEY_TEST}"
mkdir -p "$TEST_PROJECT_ANALYZE_OUT_DIR"
python experimental_pipeline/project_and_analyze.py \
    --target_ligands_repr_path "$PROCESSED_TARGET_LIGANDS_INPUT" \
    --simspace_csv_path "$COMPREHENSIVE_SIMSPACE_CSV" \
    --model_dir_for_projection "$MODELS_INPUT_DIR" \
    --model_name_root_for_projection "$MODEL_NAME_ROOT_PROJ" \
    --dr_method_key "$DR_KEY_TEST" \
    --dr_short_name "$DR_SHORT_NAME_TEST" \
    --simspace_dim $SIMSPACE_DIM_TEST \
    --k_for_knn "$K_FOR_KNN_TEST" \
    --output_dir "$TEST_PROJECT_ANALYZE_OUT_DIR" \
    --target_id_name "$TARGET_ID_NAME_TEST" \
    --representation_type "$REPR_TYPE_TEST" \
    --rdkit_features_list_target_str "$RDKIT_FEATURES_JSON_STRING"

DR_KEY_TEST="tsne"
DR_SHORT_NAME_TEST="t-SNE" # From config
PRECOMPUTED_TSNE_TARGET_PROJ="$TEST_SIMSPACE_OUT_DIR/all_dr/${MODEL_NAME_ROOT_PROJ}_tSNE_TARGET_PROJECTIONS.csv"
TEST_PROJECT_ANALYZE_OUT_DIR="$TEST_PROJECT_ANALYZE_OUT_DIR_BASE/$DR_KEY_TEST"
mkdir -p "$TEST_PROJECT_ANALYZE_OUT_DIR"
python experimental_pipeline/project_and_analyze.py \
    --precomputed_target_projections_path "$PRECOMPUTED_TSNE_TARGET_PROJ" \
    --simspace_csv_path "$COMPREHENSIVE_SIMSPACE_CSV" \
    --dr_method_key "$DR_KEY_TEST" \
    --dr_short_name "$DR_SHORT_NAME_TEST" \
    --simspace_dim $SIMSPACE_DIM_TEST \
    --k_for_knn "$K_FOR_KNN_TEST" \
    --output_dir "$TEST_PROJECT_ANALYZE_OUT_DIR" \
    --target_id_name "$TARGET_ID_NAME_TEST" \
    --representation_type "$REPR_TYPE_TEST" \
    --rdkit_features_list_target_str "$RDKIT_FEATURES_JSON_STRING"


# Example: Copy outputs from TEST_PROJECT_ANALYZE_OUT_DIR_BASE (and corresponding dim_opt plots if created)
# into a structure like: test_outputs/mock_orchestrator_run/run_test123/PyruvateKinaseM2_P14618/results/...
MOCK_RUN_DIR="test_outputs/mock_orchestrator_run/run_test123"
# Populate MOCK_RUN_DIR with a subset of results files (distances.csv, plots)
# from the project_and_analyze tests. Ensure the directory structure matches what
# the orchestrator would create.
# E.g., MOCK_RUN_DIR/PyruvateKinaseM2_P14618/results/features/dim_2/PCA/distances.csv
# MOCK_RUN_DIR/PyruvateKinaseM2_P14618/results/features/dim_2/PCA/scatter.png
# MOCK_RUN_DIR/PyruvateKinaseM2_P14618/results/features/dim_vs_mindist_PCA.png (example name)

TEST_REPORT_OUT_DIR="test_outputs/report_generation_tests"
mkdir -p "$TEST_REPORT_OUT_DIR"

python reporting/generate_latex_report.py \
    --experiment_run_dir "$MOCK_RUN_DIR" \
    --config_path "$TEST_CONFIG_PATH" \
    --output_dir "$TEST_REPORT_OUT_DIR"