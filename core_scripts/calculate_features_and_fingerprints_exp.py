import os
import logging
import gc
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from mordred import Calculator, descriptors as mordred_descriptors_module

# ------------------------ Configuration & Setup ------------------------
N_JOBS_DEFAULT = -1
RDLogger.DisableLog('rdApp.*')

# --- Mordred Descriptor List and Calculator (Unchanged) ---
desc_list = [
    mordred_descriptors_module.ABCIndex.ABCIndex,
    mordred_descriptors_module.AcidBase.AcidicGroupCount,
    mordred_descriptors_module.AcidBase.BasicGroupCount,
    mordred_descriptors_module.Aromatic.AromaticAtomsCount,
    mordred_descriptors_module.AtomCount.AtomCount('Atom'),
    mordred_descriptors_module.AtomCount.AtomCount('H'),
    mordred_descriptors_module.AtomCount.AtomCount('C'),
    mordred_descriptors_module.AtomCount.AtomCount('N'),
    mordred_descriptors_module.AtomCount.AtomCount('O'),
    mordred_descriptors_module.AtomCount.AtomCount('S'),
    mordred_descriptors_module.AtomCount.AtomCount('P'),
    mordred_descriptors_module.AtomCount.AtomCount('X'),
    mordred_descriptors_module.BondCount.BondCount('any', False),
    mordred_descriptors_module.BondCount.BondCount('heavy', False),
    mordred_descriptors_module.BondCount.BondCount('single', False),
    mordred_descriptors_module.BondCount.BondCount('double', False),
    mordred_descriptors_module.BondCount.BondCount('triple', False),
    mordred_descriptors_module.BondCount.BondCount('aromatic', False),
    mordred_descriptors_module.BondCount.BondCount('multiple', False),
    mordred_descriptors_module.BondCount.BondCount('single', True),
    mordred_descriptors_module.BondCount.BondCount('double', True),
    mordred_descriptors_module.MoeType.EState_VSA(7),
    mordred_descriptors_module.GeometricalIndex.Diameter3D,
    mordred_descriptors_module.MomentOfInertia.MomentOfInertia(axis='X'),
    mordred_descriptors_module.MomentOfInertia.MomentOfInertia(axis='Y'),
    mordred_descriptors_module.MomentOfInertia.MomentOfInertia(axis='Z'),
    mordred_descriptors_module.HydrogenBond.HBondAcceptor,
    mordred_descriptors_module.HydrogenBond.HBondDonor,
    mordred_descriptors_module.Lipinski.Lipinski,
    mordred_descriptors_module.Polarizability.APol,
    mordred_descriptors_module.Polarizability.BPol,
    mordred_descriptors_module.RingCount.RingCount(None, False, False, None, None),
    mordred_descriptors_module.RingCount.RingCount(3, False, False, None, None),
    mordred_descriptors_module.RingCount.RingCount(4, False, False, None, None),
    mordred_descriptors_module.RingCount.RingCount(5, False, False, None, None),
    mordred_descriptors_module.RingCount.RingCount(6, False, False, None, None),
    mordred_descriptors_module.RingCount.RingCount(7, False, False, None, None),
    mordred_descriptors_module.RingCount.RingCount(8, False, False, None, None),
    mordred_descriptors_module.RotatableBond.RotatableBondsCount,
    mordred_descriptors_module.TopologicalIndex.Diameter,
    mordred_descriptors_module.TopologicalIndex.TopologicalShapeIndex,
    mordred_descriptors_module.CPSA.TPSA,
    mordred_descriptors_module.VdwVolumeABC.VdwVolumeABC,
    mordred_descriptors_module.Weight.Weight(True, False),
    mordred_descriptors_module.CPSA.TASA,
]
mordred_calc_instance = Calculator(desc_list, ignore_3D=True)
MORDRED_DESCRIPTOR_NAMES = [str(d) for d in mordred_calc_instance.descriptors]


# --- Helper Functions (Unchanged) ---
def compute_ecfp4_fingerprint_original_style(mol):
    if mol is None: return None
    try:
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp = mfpgen.GetFingerprint(mol)
        arr = np.zeros((2048,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception as e:
        logging.debug(f"ECFP4 computation failed: {e}")
        return None

def calculate_dipole_original_style(mol_with_conformer):
    if mol_with_conformer is None or not mol_with_conformer.GetNumConformers():
        return np.nan
    try:
        AllChem.ComputeGasteigerCharges(mol_with_conformer)
        conf = mol_with_conformer.GetConformer()
        dipole_vector = np.array([0.0, 0.0, 0.0])
        for atom in mol_with_conformer.GetAtoms():
            pos = np.array(conf.GetAtomPosition(atom.GetIdx()))
            charge_str = atom.GetProp('_GasteigerCharge')
            if not np.isfinite(float(charge_str)): return np.nan
            dipole_vector += float(charge_str) * pos
        return np.linalg.norm(dipole_vector) * 4.80320427
    except Exception as e:
        logging.debug(f"Dipole moment calculation failed: {e}")
        return np.nan

def _process_smiles_original_style_worker(smiles_string):
    """Worker function for parallel processing. Calculates all descriptors for a single SMILES."""
    nan_mordred_descs = {name: np.nan for name in MORDRED_DESCRIPTOR_NAMES}
    if not smiles_string or pd.isna(smiles_string):
        return smiles_string, {"dipole": np.nan, "descriptors": nan_mordred_descs, "fingerprint": None}

    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        return smiles_string, {"dipole": np.nan, "descriptors": nan_mordred_descs, "fingerprint": None}
    
    fp_array = compute_ecfp4_fingerprint_original_style(mol)

    mol_for_3d = Chem.AddHs(mol)
    mol_with_conformer = None
    if AllChem.EmbedMolecule(mol_for_3d, randomSeed=42) != -1:
        try:
            AllChem.UFFOptimizeMolecule(mol_for_3d)
            mol_with_conformer = mol_for_3d
        except RuntimeError:
             pass # UFF optimization can fail; proceed with unoptimized conformer

    dipole_val = calculate_dipole_original_style(mol_with_conformer)

    descriptors_dict = nan_mordred_descs.copy()
    try:
        desc_results = mordred_calc_instance(mol_with_conformer if mol_with_conformer else mol)
        temp_desc_dict = desc_results.asdict()
        for k, v in temp_desc_dict.items():
            str_k = str(k)
            if str_k in descriptors_dict and isinstance(v, (int, float)) and np.isfinite(v):
                descriptors_dict[str_k] = float(v)
    except Exception:
        pass # Keep NaNs if Mordred fails

    return smiles_string, {"dipole": dipole_val, "descriptors": descriptors_dict, "fingerprint": fp_array}


# --- Main Processing Function (Refactored to remove cache) ---
def process_input_file_orig_style(input_csv_path, output_dir, representation_type, file_label, n_jobs):
    logging.info(f"Processing file: {input_csv_path} for {representation_type}")
    try:
        df = pd.read_csv(input_csv_path, low_memory=False)
    except Exception as e:
        logging.error(f"Error reading input file {input_csv_path}: {e}")
        return
        
    if 'SMILES' not in df.columns:
        logging.error(f"'SMILES' column not found in {input_csv_path}. Cannot process.")
        return
    
    # 1. Identify unique SMILES to avoid redundant calculations
    unique_smiles = df["SMILES"].dropna().unique().tolist()
    logging.info(f"Found {len(unique_smiles)} unique SMILES to process.")

    # 2. Perform parallel calculations on the unique set of SMILES
    parallel_results = Parallel(n_jobs=n_jobs)(
        delayed(_process_smiles_original_style_worker)(s) for s in tqdm(unique_smiles, desc="Calculating descriptors/fps")
    )
    
    # 3. Store results in a dictionary for fast lookup
    results_dict = {smiles: data for smiles, data in parallel_results if smiles}
    logging.info(f"Successfully processed {len(results_dict)} SMILES.")

    # 4. Map results back to the original DataFrame to preserve order and duplicates
    output_data_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Mapping results to {os.path.basename(input_csv_path)}"):
        smiles = row.get("SMILES")
        base_row_dict = row.to_dict()

        if smiles and pd.notna(smiles) and smiles in results_dict:
            calculated_data = results_dict[smiles]
            if representation_type == "features":
                base_row_dict['DipoleMoment'] = calculated_data.get("dipole", np.nan)
                mordred_descs = calculated_data.get("descriptors", {})
                for desc_name in MORDRED_DESCRIPTOR_NAMES:
                    base_row_dict[desc_name] = mordred_descs.get(desc_name, np.nan)
            elif representation_type == "fingerprints":
                fp_array = calculated_data.get("fingerprint")
                if fp_array is not None:
                    for i, bit in enumerate(fp_array):
                        base_row_dict[f"fp_{i}"] = int(bit)
                else:
                    for i in range(2048):
                        base_row_dict[f"fp_{i}"] = np.nan
        else: # Handle SMILES that are null or failed processing
            if representation_type == "features":
                base_row_dict['DipoleMoment'] = np.nan
                for desc_name in MORDRED_DESCRIPTOR_NAMES:
                    base_row_dict[desc_name] = np.nan
            elif representation_type == "fingerprints":
                for i in range(2048):
                    base_row_dict[f"fp_{i}"] = np.nan
        
        output_data_list.append(base_row_dict)

    output_df = pd.DataFrame(output_data_list)
    output_filename = f"{file_label}_{representation_type}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        output_df.to_csv(output_path, index=False)
        logging.info(f"Saved {representation_type} to {output_path} ({len(output_df)} records)")
    except Exception as e:
        logging.error(f"Error saving output file {output_path}: {e}")
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute features/fingerprints for a target ligands input CSV.")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV (target ligands with 'SMILES').")
    parser.add_argument("--output_dir", required=True, help="Directory to save output CSV.")
    parser.add_argument("--representation_type", required=True, choices=["features", "fingerprints"])
    parser.add_argument("--file_label", required=True, help="Label for output filename.")
    parser.add_argument("--n_jobs", type=int, default=N_JOBS_DEFAULT)
    
    args = parser.parse_args()
        
    # No cache to load, just run the processing function directly
    process_input_file_orig_style(
        args.input_csv, args.output_dir, args.representation_type, 
        args.file_label, args.n_jobs
    )
    logging.info("Target ligand feature/fingerprint calculation script finished.")