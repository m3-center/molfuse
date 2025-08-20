import os
import pickle
import logging
import gc
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from joblib import Parallel, delayed

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem # For EmbedMolecule, UFFOptimizeMolecule, ComputeGasteigerCharges
from rdkit.Chem import rdFingerprintGenerator
from mordred import Calculator, descriptors as mordred_descriptors_module

# ------------------------ Configuration & Setup ------------------------
CACHE_DIR = "temp/exp_cache" 
N_JOBS_DEFAULT = -1

os.makedirs(CACHE_DIR, exist_ok=True)
RDLogger.DisableLog('rdApp.*') 

# Global cache variable
molecule_cache = {}
cache_file_path = os.path.join(CACHE_DIR, "exp_target_ligand_mol_cache_orig_style.pkl") # New cache file name

# --- Mordred Descriptor List and Calculator (as per your snippet) ---
# Note: Diameter3D and MomentOfInertia are 3D descriptors.
# If ignore_3D=True, they will return errors or NaNs unless 3D conformers are explicitly handled by Mordred for them.
# The original _process_smiles generates a 3D conformer, so they *might* work if Mordred uses it.
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
    mordred_descriptors_module.AtomCount.AtomCount('X'), # Halogens
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
    mordred_descriptors_module.GeometricalIndex.Diameter3D, # 3D
    mordred_descriptors_module.MomentOfInertia.MomentOfInertia(axis='X'), # 3D
    mordred_descriptors_module.MomentOfInertia.MomentOfInertia(axis='Y'), # 3D
    mordred_descriptors_module.MomentOfInertia.MomentOfInertia(axis='Z'), # 3D
    mordred_descriptors_module.HydrogenBond.HBondAcceptor,
    mordred_descriptors_module.HydrogenBond.HBondDonor,
    mordred_descriptors_module.Lipinski.Lipinski,
    mordred_descriptors_module.Polarizability.APol,
    mordred_descriptors_module.Polarizability.BPol,
    mordred_descriptors_module.RingCount.RingCount(None, False, False, None, None), # nRing
    mordred_descriptors_module.RingCount.RingCount(3, False, False, None, None),
    mordred_descriptors_module.RingCount.RingCount(4, False, False, None, None),
    mordred_descriptors_module.RingCount.RingCount(5, False, False, None, None),
    mordred_descriptors_module.RingCount.RingCount(6, False, False, None, None),
    mordred_descriptors_module.RingCount.RingCount(7, False, False, None, None),
    mordred_descriptors_module.RingCount.RingCount(8, False, False, None, None),
    mordred_descriptors_module.RotatableBond.RotatableBondsCount,
    mordred_descriptors_module.TopologicalIndex.Diameter,
    mordred_descriptors_module.TopologicalIndex.TopologicalShapeIndex,
    mordred_descriptors_module.CPSA.TPSA, # RDKit also has TPSA, ensure consistency if mixing
    mordred_descriptors_module.VdwVolumeABC.VdwVolumeABC,
    mordred_descriptors_module.Weight.Weight(True, False), # RDKit also has MolWt
    mordred_descriptors_module.CPSA.TASA,
]
# Initialize Mordred Calculator using the defined list
# ignore_3D=False because some listed descriptors are 3D and _process_smiles generates 3D conformer.
mordred_calc_instance = Calculator(desc_list, ignore_3D=True) 
# Get string names of descriptors for column headers and NaN initialization
MORDRED_DESCRIPTOR_NAMES = [str(d) for d in mordred_calc_instance.descriptors]


# ------------------------ Helper Functions (from your snippets) ------------------------
def compute_ecfp4_fingerprint_original_style(mol):
    """Compute ECFP4 (radius=2, nBits=2048) fingerprint as a numpy array."""
    if mol is None: return None
    try:
        # For ECFP4, typically use the mol without explicit Hs from SMILES
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp = mfpgen.GetFingerprint(mol) 
        arr = np.zeros((2048,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception as e:
        logging.debug(f"ECFP4 computation failed for a SMILES: {e}")
        return None

def calculate_dipole_original_style(mol_with_conformer):
    """Calculate dipole moment in Debye for an RDKit molecule with a conformer."""
    if mol_with_conformer is None or not mol_with_conformer.GetNumConformers():
        return np.nan
    try:
        AllChem.ComputeGasteigerCharges(mol_with_conformer)
        conf = mol_with_conformer.GetConformer()
        dipole_vector = np.array([0.0, 0.0, 0.0]) # Renamed to avoid conflict
        for atom in mol_with_conformer.GetAtoms():
            idx = atom.GetIdx()
            pos_np_vec = np.array(conf.GetAtomPosition(idx)) # Renamed
            charge_prop_val = atom.GetProp('_GasteigerCharge') # Renamed
            # Robust check for charge validity
            if isinstance(charge_prop_val, str) and (charge_prop_val.lower() in ['nan', 'inf', '-inf']):
                return np.nan
            try:
                charge_val = float(charge_prop_val) # Renamed
            except ValueError:
                return np.nan # Cannot convert charge to float
            dipole_vector += charge_val * pos_np_vec
        return np.linalg.norm(dipole_vector) * 4.80320427 # Your original factor
    except Exception as e:
        # logging.debug to avoid flooding logs for common issues like failed Gasteiger charges
        logging.debug(f"Dipole moment calculation failed for a SMILES: {e}")
        return np.nan

def _process_smiles_original_style_worker(smiles_string):
    """
    Worker function for parallel processing, matching your original _process_smiles.
    """
    nan_mordred_descs = {name: np.nan for name in MORDRED_DESCRIPTOR_NAMES}
    if not smiles_string or pd.isna(smiles_string):
        return smiles_string, {"dipole": np.nan, "descriptors": nan_mordred_descs, "fingerprint": None}

    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        logging.debug(f"Could not parse SMILES: {smiles_string}")
        return smiles_string, {"dipole": np.nan, "descriptors": nan_mordred_descs, "fingerprint": None}
    
    # Create a mol for fingerprinting (typically without explicit Hs unless canonicalization needs them)
    mol_for_fp = Chem.MolFromSmiles(smiles_string) # Fresh mol
    fp_array = compute_ecfp4_fingerprint_original_style(mol_for_fp) # Pass the correct mol

    # Prepare mol for 3D dependent calculations (dipole, 3D Mordred)
    mol_for_3d = Chem.AddHs(Chem.MolFromSmiles(smiles_string)) # Fresh mol, add Hs
    if mol_for_3d is None: # Should not happen if initial mol was valid
         return smiles_string, {"dipole": np.nan, "descriptors": nan_mordred_descs, "fingerprint": fp_array}

    # Embed and Optimize for 3D
    embed_success = AllChem.EmbedMolecule(mol_for_3d, randomSeed=42, maxAttempts=1000)
    if embed_success == -1: # Try with basic knowledge if first attempt fails
        embed_success = AllChem.EmbedMolecule(mol_for_3d, randomSeed=42, useBasicKnowledge=True, maxAttempts=1000)
    
    mol_with_conformer = None
    if embed_success != -1:
        try:
            AllChem.UFFOptimizeMolecule(mol_for_3d, maxIters=200)
            mol_with_conformer = mol_for_3d
        except RuntimeError: # Catch specific optimization errors like "UFFOptimizeMolecule() failed"
            logging.debug(f"UFF optimization failed for SMILES: {smiles_string}")
            # Proceed without optimized 3D for dipole, or return NaN for dipole
            mol_with_conformer = None # Or keep mol_for_3d if some conformer is better than none
    else:
        logging.debug(f"Embedding failed for SMILES: {smiles_string}")

    dipole_val = calculate_dipole_original_style(mol_with_conformer if mol_with_conformer else mol_for_3d) # Try with unoptimized if opt failed

    # Calculate Mordred descriptors
    # Mordred's ignore_3D=False means it will attempt to use the conformer if available and needed
    descriptors_dict = nan_mordred_descs.copy() # Start with NaNs
    try:
        # Pass the mol that has undergone H addition and potential 3D embedding.
        # If Mordred needs a very specific type of mol object, this might need adjustment.
        # Usually, a mol object with a conformer is what it would use for 3D descriptors.
        desc_obj_calc = mordred_calc_instance(mol_with_conformer if mol_with_conformer else mol_for_3d)
        temp_desc_dict = desc_obj_calc.asdict()
        for k, v in temp_desc_dict.items():
             # Ensure keys are strings (as in MORDRED_DESCRIPTOR_NAMES) and values are float-compatible
            str_k = str(k)
            if str_k in descriptors_dict:
                if isinstance(v, (int, float)) and np.isfinite(v):
                    descriptors_dict[str_k] = float(v)
                # Mordred can return error objects as values
                elif not isinstance(v, (str, Chem.Mol, type(None))) and not hasattr(v, 'message'): # Check if not an error object
                     try:
                         descriptors_dict[str_k] = float(v)
                     except:
                         pass # Keep NaN
    except Exception as e:
        logging.debug(f"Mordred descriptor calculation failed for SMILES {smiles_string}: {e}")
        # descriptors_dict remains NaNs for all

    return smiles_string, {"dipole": dipole_val, "descriptors": descriptors_dict, "fingerprint": fp_array}


# --- Caching Functions (similar to previous experimental version) ---
def load_global_mol_cache_orig_style():
    global molecule_cache
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, "rb") as f: molecule_cache = pickle.load(f)
            logging.info(f"Loaded {len(molecule_cache)} items from cache: {cache_file_path}")
        except Exception as e:
            logging.error(f"Error loading cache '{cache_file_path}': {e}. Starting empty.")
            molecule_cache = {}

def save_global_mol_cache_orig_style():
    global molecule_cache
    try:
        with open(cache_file_path, "wb") as f: pickle.dump(molecule_cache, f)
        logging.info(f"Saved {len(molecule_cache)} items to cache: {cache_file_path}")
    except Exception as e: logging.error(f"Error saving cache '{cache_file_path}': {e}")

def update_molecule_cache_orig_style(smiles_list, n_jobs):
    global molecule_cache
    new_smiles = sorted(list(set(s for s in smiles_list if s and pd.notna(s) and s not in molecule_cache)))
    
    if not new_smiles:
        logging.info("No new SMILES to process for cache.")
        return

    logging.info(f"Processing {len(new_smiles)} new unique SMILES for cache update...")
    # Dynamic batch size, good for varying n_jobs and list sizes
    batch_size = max(100, len(new_smiles) // (abs(n_jobs) if n_jobs != 0 else os.cpu_count() or 1) // 2 + 1) 
    batch_size = min(batch_size, 500) # Cap batch size

    for i in range(0, len(new_smiles), batch_size):
        batch = new_smiles[i:i+batch_size]
        logging.info(f"Processing cache batch {i//batch_size + 1}/{ (len(new_smiles) -1 )//batch_size + 1} (size {len(batch)})...")
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_smiles_original_style_worker)(s) for s in tqdm(batch, desc="Calculating descriptors/fps", leave=False)
        )
        for smiles_key, res_dict in results:
            if smiles_key: molecule_cache[smiles_key] = res_dict
        
        save_global_mol_cache_orig_style() 
        gc.collect()

# --- Main Processing Function ---
def process_input_file_orig_style(input_csv_path, output_dir, representation_type, file_label, n_jobs):
    logging.info(f"Processing file: {input_csv_path} for {representation_type} (original style)")
    try:
        df = pd.read_csv(input_csv_path, low_memory=False)
    except Exception as e:
        logging.error(f"Error reading input file {input_csv_path}: {e}")
        return
        
    if 'SMILES' not in df.columns:
        logging.error(f"'SMILES' column not found in {input_csv_path}. Cannot process.")
        return
    
    unique_smiles_in_file = df["SMILES"].dropna().unique().tolist()
    update_molecule_cache_orig_style(unique_smiles_in_file, n_jobs)

    output_data_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Applying cache to {os.path.basename(input_csv_path)}"):
        smiles = row.get("SMILES")
        base_row_dict = row.to_dict() # Keep all original columns

        if smiles and pd.notna(smiles) and smiles in molecule_cache:
            cached_data = molecule_cache[smiles]
            if representation_type == "features":
                base_row_dict['DipoleMoment'] = cached_data.get("dipole", np.nan)
                mordred_descs_from_cache = cached_data.get("descriptors", {})
                for desc_name in MORDRED_DESCRIPTOR_NAMES: # Ensure all expected Mordred cols are present
                    base_row_dict[desc_name] = mordred_descs_from_cache.get(desc_name, np.nan)
            elif representation_type == "fingerprints":
                fp_array = cached_data.get("fingerprint")
                if fp_array is not None:
                    for i, bit in enumerate(fp_array): base_row_dict[f"fp_{i}"] = int(bit)
                else: # Fingerprint could not be calculated
                    for i in range(2048): base_row_dict[f"fp_{i}"] = np.nan
            output_data_list.append(base_row_dict)
        else: 
            if representation_type == "features":
                base_row_dict['DipoleMoment'] = np.nan
                for desc_name in MORDRED_DESCRIPTOR_NAMES: base_row_dict[desc_name] = np.nan
            elif representation_type == "fingerprints":
                for i in range(2048): base_row_dict[f"fp_{i}"] = np.nan
            output_data_list.append(base_row_dict)
            if smiles and pd.notna(smiles):
                 logging.warning(f"SMILES '{smiles}' processed but not found in cache. Resulting data will be NaN.")

    output_df = pd.DataFrame(output_data_list)
    
    output_filename = f"{file_label}_{representation_type}.csv" # Consistent naming
    output_path = os.path.join(output_dir, output_filename)
    try:
        output_df.to_csv(output_path, index=False)
        logging.info(f"Saved {representation_type} (original style) to {output_path} ({len(output_df)} records)")
    except Exception as e:
        logging.error(f"Error saving output file {output_path}: {e}")
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute features/fingerprints (original style) for TARGET LIGANDS input CSV.")
    parser.add_argument("--input_csv", required=True, help="Path to input CSV (target ligands with 'SMILES').")
    parser.add_argument("--output_dir", required=True, help="Directory to save output CSV.")
    parser.add_argument("--representation_type", required=True, choices=["features", "fingerprints"])
    parser.add_argument("--file_label", required=True, help="Label for output filename.")
    parser.add_argument("--n_jobs", type=int, default=N_JOBS_DEFAULT)
    # No --rdkit_features_list_target_str needed as feature set is fixed by `desc_list` internally
    
    args = parser.parse_args()
        
    load_global_mol_cache_orig_style()
    process_input_file_orig_style(args.input_csv, args.output_dir, args.representation_type, 
                                  args.file_label, args.n_jobs)
    logging.info("Target ligand feature/fingerprint calculation (original style) script finished.")