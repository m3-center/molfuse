# MolFuSE GUI: Production Candidate Scoring Tool

**Version:** 1.0  
**Purpose:** Interactive tool for scoring candidate molecules against trained MolFuSE similarity models.

---

## Overview

The MolFuSE GUI is a production-ready web interface for projecting and scoring candidate molecules using pre-trained molecular function similarity models. Unlike the Phase 1-4 analysis tools (which are for experimental research), this GUI is designed for practical screening applications.

**Key Features:**
- 📊 **Model Browser**: Select from trained models by molecular function (KW category)
- 🧪 **Candidate Input**: Upload CSV or paste SMILES strings
- 🤖 **Automatic Descriptor Computation**: RDKit 2D features computed on-the-fly
- 📈 **Real-Time Scoring**: 1-NN similarity scoring against MF cloud
- 🔬 **Interactive Visualization**: 2D scatter plots, score distributions, ROC/PR curves
- 💾 **Export Results**: Ranked candidate list with nearest neighbor info

**Performance:**
- Optimized for 1-1,000 candidates with real-time response
- For >10k candidates, use CLI alternative (see below)

---

## Installation

### Prerequisites

```bash
# Required Python packages (if not already installed)
pip install dash dash-bootstrap-components plotly pandas numpy scikit-learn rdkit joblib
```

### Verify Installation

```bash
# Test imports
python -c "from molfuse.gui.app import main; print('GUI ready!')"
```

---

## Quick Start: Demo Case

A lightweight demo workspace is provided for testing without large memory requirements.

### 1. Generate Demo Data

```bash
python scripts/generate_gui_demo.py
```

This creates:
- `demo/workspace/` - Minimal Phase 1 run (100 MF, 1000 ZINC, 50 actives)
- `demo/data/example_candidates.csv` - 10 example molecules

### 2. Launch GUI

```bash
python -m molfuse.gui.app --workspace demo/workspace
```

Or launch without pre-configured workspace:

```bash
python -m molfuse.gui.app
```

Then enter workspace path in the GUI.

### 3. Open Browser

Navigate to: **http://127.0.0.1:8050**

### 4. Walkthrough

1. **Select Model:**
   - KW Category: `KW-0049_Antioxidant`
   - Model Run: `demo_antioxidant_pca_2d | PCA 2D | EF@1%: 5.00`

2. **Input Candidates:**
   - Click "Upload CSV" tab
   - Upload `demo/data/example_candidates.csv`
   - Wait for descriptor computation confirmation

3. **Score:**
   - Click "Score Candidates" button
   - View results: scatter plot, score distribution, top candidates table

4. **Export:**
   - Click "Download Scored Candidates CSV"
   - Opens ranked list with scores, distances, nearest MF neighbors

---

## Production Usage

### With Existing Workspace

If you've already run Phase 1/4 experiments:

```bash
python -m molfuse.gui.app --workspace /path/to/experiment_workspace_v4
```

The GUI will scan for all valid models in `phase1/` and `phase4/` directories.

### Model Selection Criteria

The GUI auto-detects "appropriate" models by:
1. **Target Accession**: Matches protein (e.g., P00519)
2. **Molecular Function (KW)**: User selects from dropdown (e.g., `KW-0505_Motor_protein`)
3. **Representation**: Features only (fingerprints not yet supported in GUI v1.0)
4. **Performance**: Models ranked by EF@1% (best auto-selected)

### Input Formats

**Option 1: CSV Upload**
- Required column: `SMILES`
- Optional columns: `Compound ChEMBL ID`, custom metadata (preserved in output)

Example `candidates.csv`:
```csv
SMILES,Compound ChEMBL ID,Source
CC(C)Cc1ccc(cc1)C(C)C(O)=O,CHEMBL123,Vendor_A
CCOc1ccc2nc(sc2c1)S(=O)(=O)N,CHEMBL456,Vendor_B
```

**Option 2: Text Input**
- One SMILES per line
- No header required

```
CC(C)Cc1ccc(cc1)C(C)C(O)=O
CCOc1ccc2nc(sc2c1)S(=O)(=O)N
c1ccc2c(c1)ccc3c2ccc4c3cccc4
```

### Output Columns

Downloaded CSV includes:
- **SMILES**: Candidate SMILES string
- **score**: Similarity score (-distance; higher = better)
- **rank**: Rank by score (1 = best)
- **distance**: Min distance to MF cloud in embedding space
- **z0, z1, ...**: Embedding coordinates
- **nearest_mf_smiles**: SMILES of nearest MF neighbor
- **nearest_mf_activity_type**: Activity type (IC50, Ki, Kd, EC50)
- **nearest_mf_activity_value_nM**: Activity value in nM
- **nearest_mf_accession**: Target protein accession
- *[User-provided metadata columns]*

---

## Visualization Guide

### 2D Scatter Plot

- **Blue points**: MF cloud (training set)
- **Purple points**: ZINC decoys (if available)
- **Red points**: Known actives (if available)
- **Green points**: Your candidates (colored by score)

**Projection Methods:**
- If model is 2D: plots z0 vs z1 directly
- If model is >2D (e.g., 10D UMAP): applies PCA projection to 2D for visualization only (scoring still uses full dimensionality)

### Score Distribution

Histogram showing where your candidates fall relative to MF cloud and actives.

### ROC/PR Curves

Only displayed if actives are available in the loaded model run.

---

## Performance Notes

### Real-Time Scoring (GUI)

**Capacity:** 1-1,000 candidates  
**Typical Response Time:**
- 100 candidates: ~5-10 seconds
- 1,000 candidates: ~30-60 seconds

**Bottlenecks:**
1. Descriptor computation (RDKit): ~0.05s per molecule
2. Projection through DR model: ~0.01s per molecule
3. 1-NN scoring: <0.001s per query (efficient KDTree)

### CLI Alternative (Large Batches)

For >10,000 candidates, use the CLI scorer (coming in v1.1):

```bash
# Future CLI interface
python -m molfuse.cli.score \
    --workspace experiment_workspace_v4 \
    --model_run phase1_best_umap_10d \
    --candidates large_candidates.csv \
    --output scored_output.csv
```

**Advantages:**
- Batch processing with progress bars
- Parallel descriptor computation
- Lower memory footprint (streaming I/O)
- HPC cluster submission support

---

## Troubleshooting

### "No models found in workspace"

**Cause:** Workspace doesn't contain valid Phase 1/4 runs.

**Solution:**
1. Verify workspace path: `ls /path/to/workspace/phase1/`
2. Check for required artifacts:
   ```
   phase1/<run_name>/artifacts/scaler.joblib
   phase1/<run_name>/artifacts/pca_model.joblib (or umap_model.joblib)
   phase1/<run_name>/artifacts/embedding_mf.csv
   ```
3. Re-run Phase 1 if artifacts are missing

### "Candidates missing expected features"

**Cause:** Descriptor computation failed or model expects features not computed by RDKit.

**Solution:**
1. Check SMILES validity (GUI will report parse failures)
2. Verify model's feature set: GUI automatically matches scaler's expected features
3. If using custom descriptors, pre-compute and upload CSV with all required columns

### "Score button disabled"

**Causes:**
- No model selected
- No candidates uploaded
- Descriptor computation failed

**Solution:** Check upload status message for specific error.

### Memory Issues

**Symptoms:** GUI crashes or browser freezes

**Solutions:**
1. Reduce candidate count (batch if >1000)
2. Close unused browser tabs
3. Use CLI for large batches
4. Increase system RAM allocation

---

## Advanced Configuration

### Launching on Different Port

```bash
python -m molfuse.gui.app --workspace demo/workspace --port 8080
```

### Enabling Remote Access

```bash
python -m molfuse.gui.app --workspace demo/workspace --host 0.0.0.0 --port 8050
```

**⚠️ Security Warning:** Only enable remote access on trusted networks. No authentication is implemented in v1.0.

### Debug Mode

```bash
python -m molfuse.gui.app --workspace demo/workspace --debug
```

Enables:
- Hot reload on code changes
- Detailed error tracebacks in browser
- Verbose logging

---

## Comparison: GUI vs CLI

| Feature | GUI | CLI (Phase 1-4) |
|---------|-----|-----------------|
| **Use Case** | Scoring new candidates | Training models, experiments |
| **Capacity** | 1-1,000 molecules | Unlimited (100k+) |
| **Response Time** | Real-time (<1 min) | Batch (minutes to hours) |
| **Visualization** | Interactive plots | Static reports |
| **Model Selection** | Dropdown by KW | Config JSON |
| **Descriptor Computation** | Automatic (RDKit) | Pre-computed CSVs |
| **HPC Support** | No | Yes (SLURM) |
| **Reproducibility** | Interactive session | Fully logged |

**Recommendation:**
- Use **GUI** for exploratory screening (<1000 candidates)
- Use **CLI** for production-scale screening (>10k candidates)

---

## Future Enhancements (Roadmap)

### v1.1 (Planned)
- [ ] CLI scorer module for large batches
- [ ] Fingerprint representation support
- [ ] Batch upload (multiple CSV files)
- [ ] Custom descriptor upload (bypass RDKit computation)

### v1.2 (Planned)
- [ ] Multi-model comparison view
- [ ] Activity prediction (calibrated scoring)
- [ ] Diversity analysis (cluster candidates)
- [ ] Export molecular structures (SDF)

### v2.0 (Future)
- [ ] Model training interface (Phase 1 GUI)
- [ ] User authentication & session management
- [ ] API for programmatic access
- [ ] Docker containerization

---

## Citation

If you use MolFuSE GUI in your research, please cite:

> [Your Publication]  
> MolFuSE: Molecular Function Similarity for Virtual Screening  
> [Journal, Year]

---

## Support

**Issues:** https://github.com/alexander-hagg/UMMBAS_screening_experiments/issues  
**Documentation:** `README_V4_MOLFUSE.md` (pipeline details)  
**Contact:** [Your Contact Info]

---

## License

[Your License Here]

---

## Changelog

### v1.0.0 (2025-11-20)
- Initial release
- Model browser with KW category filtering
- CSV and text SMILES input
- Real-time descriptor computation (RDKit)
- 1-NN scoring with nearest neighbor lookup
- Interactive 2D scatter plots (with PCA projection for >2D models)
- Score distribution histograms
- Top candidates table with structure rendering
- CSV export with full metadata
- Lightweight demo case (Antioxidant/P00441)
