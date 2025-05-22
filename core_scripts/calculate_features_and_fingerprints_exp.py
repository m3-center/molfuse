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
from rdkit.Chem import AllChem, Descriptors as RDKitDescriptors # Added RDKitDescriptors
from rdkit.Chem import rdFingerprintGenerator
from mordred import Calculator, descriptors

# ------------------------ Configuration Defaults (for CLI) ------------------------
DEFAULT_OUTPUT_DIR = "temp/exp_features_fingerprints" # For standalone testing
CACHE_DIR = "temp/exp_cache" # Centralized cache for experimental runs
N_JOBS_DEFAULT = -1

os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# ------------------------ Mordred Descriptor Calculator ------------------------
# Using the same list as in the original script
mordred_desc_list = [
    descriptors.ABCIndex.ABCIndex, descriptors.AcidBase.AcidicGroupCount,
    descriptors.AcidBase.BasicGroupCount, descriptors.Aromatic.AromaticAtomsCount,
    descriptors.AtomCount.AtomCount('Atom'), descriptors.AtomCount.AtomCount('H'),
    descriptors.AtomCount.AtomCount('C'), descriptors.AtomCount.AtomCount('N'),
    descriptors.AtomCount.AtomCount('O'), descriptors.AtomCount.AtomCount('S'),
    descriptors.AtomCount.AtomCount('P'), descriptors.AtomCount.AtomCount('X'),
    descriptors.BondCount.BondCount('any'), # Explicitly 'any'
    descriptors.BondCount.BondCount('single', True), descriptors.BondCount.BondCount('double', True),
    descriptors.BondCount.BondCount('triple', True), descriptors.BondCount.BondCount('aromatic', True),
    descriptors.BondCount.BondCount('misc', True), # Added misc
    descriptors.MoeType.EState_VSA(7),
    # descriptors.GeometricalIndex.Diameter3D, # Removing 3D for consistency with ignore_3D=True
    # descriptors.MomentOfInertia.MomentOfInertia(axis='X'), # Removing 3D
    # descriptors.MomentOfInertia.MomentOfInertia(axis='Y'), # Removing 3D
    # descriptors.MomentOfInertia.MomentOfInertia(axis='Z'), # Removing 3D
    descriptors.HydrogenBond.HBondAcceptor, descriptors.HydrogenBond.HBondDonor,
    descriptors.Lipinski.Lipinski, descriptors.Polarizability.APol,
    descriptors.Polarizability.BPol,
    descriptors.RingCount.RingCount(None, False, False, None, None), # nRing
    descriptors.RingCount.RingCount(3, False, False, None, None),   # n3Ring
    descriptors.RingCount.RingCount(4, False, False, None, None),   # n4Ring
    descriptors.RingCount.RingCount(5, False, False, None, None),   # n5Ring
    descriptors.RingCount.RingCount(6, False, False, None, None),   # n6Ring
    descriptors.RingCount.RingCount(7, False, False, None, None),   # n7Ring
    descriptors.RingCount.RingCount(8, False, False, None, None),   # n8Ring
    descriptors.RotatableBond.RotatableBondsCount,
    descriptors.TopologicalIndex.Diameter, descriptors.TopologicalIndex.TopologicalShapeIndex,
    # descriptors.CPSA.TPSA, # TPSA is calculated by RDKit below
    descriptors.VdwVolumeABC.VdwVolumeABC, # Vabc
    # descriptors.Weight.Weight(True, False), # MW is calculated by RDKit below
    # descriptors.CPSA.TASA, # TASA
]
mordred_calc = Calculator(mordred_desc_list, ignore_3D=True)

# ------------------------ RDKit Feature List (from GUI for reference/consistency) ------------------------
# This list will be used if 'mordred' is not the sole source of features
# and we want to match the GUI's feature set.
RDKIT_FEATURES_FOR_GUI = {
    'DipoleMoment': lambda mol: calculate_dipole(mol), # Needs 3D conformer
    'ABC': lambda mol: mordred_calc.descriptors[0](mol) if mol else np.nan, # Example: ABCIndex
    'nAcid': lambda mol: mordred_calc.descriptors[1](mol) if mol else np.nan, # AcidicGroupCount
    'nBase': lambda mol: mordred_calc.descriptors[2](mol) if mol else np.nan, # BasicGroupCount
    'nAromAtom': lambda mol: mordred_calc.descriptors[3](mol) if mol else np.nan, # AromaticAtomsCount
    'nAtom': lambda mol: mordred_calc.descriptors[4](mol) if mol else np.nan, # AtomCount('Atom')
    'nH': lambda mol: mordred_calc.descriptors[5](mol) if mol else np.nan,
    'nC': lambda mol: mordred_calc.descriptors[6](mol) if mol else np.nan,
    'nN': lambda mol: mordred_calc.descriptors[7](mol) if mol else np.nan,
    'nO': lambda mol: mordred_calc.descriptors[8](mol) if mol else np.nan,
    'nS': lambda mol: mordred_calc.descriptors[9](mol) if mol else np.nan,
    'nP': lambda mol: mordred_calc.descriptors[10](mol) if mol else np.nan,
    'nX': lambda mol: mordred_calc.descriptors[11](mol) if mol else np.nan, # Halogen count
    'nBonds': lambda mol: RDKitDescriptors.HeavyAtomCount(mol) -1 + sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1) if mol else np.nan, # Approx, RDKit doesn't have simple total bond count
    'nBondsO': lambda mol: sum(1 for b in mol.GetBonds() if b.GetBondTypeAsDouble() == 0) if mol else np.nan, # Other
    'nBondsS': lambda mol: sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.rdchem.BondType.SINGLE) if mol else np.nan,
    'nBondsD': lambda mol: sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.rdchem.BondType.DOUBLE) if mol else np.nan,
    'nBondsT': lambda mol: sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.rdchem.BondType.TRIPLE) if mol else np.nan,
    'nBondsA': lambda mol: sum(1 for b in mol.GetBonds() if b.GetBondType() == Chem.rdchem.BondType.AROMATIC) if mol else np.nan,
    'nBondsM': lambda mol: np.nan, # Placeholder for 'misc' - define if needed
    'nBondsKS': lambda mol: np.nan, # Placeholder for specific bond counts if needed
    'nBondsKD': lambda mol: np.nan, # Placeholder for specific bond counts if needed
    'EState_VSA7': lambda mol: mordred_calc.descriptors[15](mol) if mol else np.nan, # EState_VSA(7)
    'nHBAcc': lambda mol: RDKitDescriptors.NumHAcceptors(mol) if mol else np.nan,
    'nHBDon': lambda mol: RDKitDescriptors.NumHDonors(mol) if mol else np.nan,
    'Lipinski': lambda mol: RDKitDescriptors.LipinskiRuleOfFive_violationContribs(mol)[0] if mol else np.nan, # Number of violations
    'apol': lambda mol: RDKitDescriptors.MolLogP(mol) if mol else np.nan, # Using MolLogP as 'apol' (often AlogP)
    'bpol': lambda mol: RDKitDescriptors.TPSA(mol) if mol else np.nan, # Using TPSA as 'bpol' (often related to polar surface area)
    'nRing': lambda mol: RDKitDescriptors.RingCount(mol) if mol else np.nan,
    'n3Ring': lambda mol: sum(1 for r in mol.GetRingInfo().AtomRings() if len(r) == 3) if mol else np.nan,
    'n4Ring': lambda mol: sum(1 for r in mol.GetRingInfo().AtomRings() if len(r) == 4) if mol else np.nan,
    'n5Ring': lambda mol: sum(1 for r in mol.GetRingInfo().AtomRings() if len(r) == 5) if mol else np.nan,
    'n6Ring': lambda mol: sum(1 for r in mol.GetRingInfo().AtomRings() if len(r) == 6) if mol else np.nan,
    'n7Ring': lambda mol: sum(1 for r in mol.GetRingInfo().AtomRings() if len(r) == 7) if mol else np.nan,
    'n8Ring': lambda mol: sum(1 for r in mol.GetRingInfo().AtomRings() if len(r) == 8) if mol else np.nan,
    'nRot': lambda mol: RDKitDescriptors.NumRotatableBonds(mol) if mol else np.nan,
    'Diameter': lambda mol: mordred_calc.descriptors[26](mol) if mol else np.nan, # Topological Diameter
    'TopoShapeIndex': lambda mol: mordred_calc.descriptors[27](mol) if mol else np.nan, # TopologicalShapeIndex
    'Vabc': lambda mol: mordred_calc.descriptors[28](mol) if mol else np.nan, # VdwVolumeABC
    'MW': lambda mol: RDKitDescriptors.MolWt(mol) if mol else np.nan
}
# This is the final list of feature names that this script will aim to produce for 'features' mode.
# It should align with RDKIT_FEATURES_LIST in gui_similarity.py for consistency if custom uploads are used.
RDKIT_FEATURES_LIST_TARGET = [
    'DipoleMoment','ABC','nAcid','nBase','nAromAtom','nAtom','nH','nC','nN','nO','nS',
    'nP','nX','nBonds','nBondsO','nBondsS','nBondsD','nBondsT','nBondsA','nBondsM',
    'nBondsKS','nBondsKD','EState_VSA7','nHBAcc','nHBDon','Lipinski','apol','bpol',
    'nRing','n3Ring','n4Ring','n5Ring','n6Ring','n7Ring','n8Ring','nRot','Diameter',
    'TopoShapeIndex','Vabc','MW'
]


# ------------------------ Helper Functions (adapted) ------------------------
def compute_ecfp4_fingerprint(mol):
    if mol is None: return None
    try:
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp = mfpgen.GetFingerprint(mol)
        arr = np.zeros((2048,), dtype=np.int8) # Using int8 to save memory if bits are 0/1
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception as e:
        logging.debug(f"ECFP4 computation failed: {e}")
        return None

def calculate_dipole(mol_3d): # Expects a mol with a 3D conformer
    if mol_3d is None or not mol_3d.GetNumConformers():
        return np.nan
    try:
        # Gasteiger charges might not be best for dipole, but it's what was in GUI.
        # For better dipoles, QM or more advanced MM charges are needed.
        AllChem.ComputeGasteigerCharges(mol_3d) 
        conf = mol_3d.GetConformer()
        dipole_vec = np.array([0.0, 0.0, 0.0])
        for atom in mol_3d.GetAtoms():
            idx = atom.GetIdx()
            pos = np.array(conf.GetAtomPosition(idx))
            charge_prop = atom.GetProp('_GasteigerCharge')
            if charge_prop == 'nan' or charge_prop == 'inf': # Handle non-numeric charges
                return np.nan
            charge = float(charge_prop)
            dipole_vec += charge * pos
        # Conversion factor from e*Angstrom to Debye (1 D = 3.33564e-30 C*m; e=1.602e-19 C; A=1e-10 m)
        # (1.602e-19 * 1e-10) / 3.33564e-30 = 4.803
        return np.linalg.norm(dipole_vec) * 4.8032 
    except Exception as e:
        logging.debug(f"Dipole moment calculation failed: {e}")
        return np.nan

def get_rdkit_and_mordred_features(mol_3d, mol_2d, target_feature_list, mordred_calculator, rdkit_gui_features_map):
    """
    Calculates a defined set of features using RDKit and Mordred.
    mol_3d is for 3D dependent features, mol_2d for others.
    """
    features = {}
    
    # Calculate Mordred features first (can be done on mol_2d if ignore_3D=True)
    mordred_values = {}
    if mol_2d:
        try:
            mordred_descs_obj = mordred_calculator(mol_2d)
            mordred_values = {str(k): float(v) if not isinstance(v, (str, type(None))) and np.isfinite(v) else np.nan for k, v in mordred_descs_obj.asdict().items()}
        except Exception as e:
            logging.debug(f"Mordred calculation failed for a SMILES: {e}")
            # Fill with NaNs for all expected Mordred descriptors
            for desc_obj in mordred_calculator.descriptors:
                 mordred_values[str(desc_obj)] = np.nan
    else: # mol_2d is None
        for desc_obj in mordred_calculator.descriptors:
            mordred_values[str(desc_obj)] = np.nan


    # Calculate RDKit features from the GUI list, using Mordred values where specified
    for feature_name in target_feature_list:
        if feature_name in rdkit_gui_features_map:
            try:
                # Select appropriate molecule (3D for dipole, 2D for most others)
                mol_for_calc = mol_3d if feature_name == "DipoleMoment" else mol_2d
                
                # Some lambda functions in rdkit_gui_features_map might refer to specific indices
                # of mordred_calc.descriptors. This needs to be robust.
                # For now, assume the lambdas are either direct RDKit calls or can handle mol_for_calc=None
                val = rdkit_gui_features_map[feature_name](mol_for_calc)
                features[feature_name] = float(val) if not isinstance(val, (str,type(None))) and np.isfinite(val) else np.nan
            except Exception as e:
                logging.debug(f"RDKit feature '{feature_name}' calculation failed: {e}")
                features[feature_name] = np.nan
        elif feature_name in mordred_values: # Fallback to pre-calculated Mordred if not in RDKIT_FEATURES_FOR_GUI map
            features[feature_name] = mordred_values[feature_name]
        else:
            # This case should ideally not happen if RDKIT_FEATURES_LIST_TARGET is well-defined
            logging.warning(f"Feature '{feature_name}' not found in RDKit/Mordred maps. Setting to NaN.")
            features[feature_name] = np.nan
            
    return features


def _process_smiles_for_cache(smiles_string):
    """
    Worker function to process a single SMILES string.
    Generates 2D and 3D RDKit mol objects, ECFP4, and a defined set of RDKit+Mordred features.
    """
    if not smiles_string or pd.isna(smiles_string):
        return smiles_string, {"mol_2d": None, "mol_3d": None, "ecfp4": None, "features": {k: np.nan for k in RDKIT_FEATURES_LIST_TARGET}}

    mol_2d = Chem.MolFromSmiles(smiles_string)
    mol_3d_prepared = None
    features = {k: np.nan for k in RDKIT_FEATURES_LIST_TARGET} # Initialize with NaNs
    ecfp4 = None

    if mol_2d:
        mol_2d = Chem.AddHs(mol_2d) # AddHs for consistency, though many 2D descs don't need them
        
        # Prepare 3D conformer for dipole and any 3D Mordred (though Mordred set to ignore_3D)
        mol_for_3d = Chem.MolFromSmiles(smiles_string) # Fresh mol for 3D
        if mol_for_3d:
            mol_for_3d_h = Chem.AddHs(mol_for_3d)
            embed_success = AllChem.EmbedMolecule(mol_for_3d_h, randomSeed=42, useRandomCoords=False, maxAttempts=1000) # Increased attempts
            if embed_success != -1:
                try:
                    AllChem.UFFOptimizeMolecule(mol_for_3d_h, maxIters=200)
                    mol_3d_prepared = mol_for_3d_h
                except Exception: # Catch specific rdkit geometry errors if possible
                    mol_3d_prepared = None # Optimization failed
            else: # Embed failed
                 mol_3d_prepared = None
        
        # Calculate features
        features = get_rdkit_and_mordred_features(mol_3d_prepared, mol_2d, RDKIT_FEATURES_LIST_TARGET, mordred_calc, RDKIT_FEATURES_FOR_GUI)
        
        # Calculate ECFP4 (on 2D mol without Hs is standard)
        mol_for_fp = Chem.MolFromSmiles(smiles_string) # Fresh molecule for fingerprinting
        ecfp4 = compute_ecfp4_fingerprint(mol_for_fp)
    
    # Clean up mol objects if they are large and not needed directly in cache value
    # For now, keeping them for potential direct use, but could be removed if cache size is an issue
    return smiles_string, {"mol_2d": mol_2d, "mol_3d": mol_3d_prepared, "ecfp4": ecfp4, "features": features}


molecule_cache = {} # Global cache for this script run
cache_file_path = os.path.join(CACHE_DIR, "exp_molecule_cache.pkl")

def load_global_cache():
    global molecule_cache
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, "rb") as f:
                molecule_cache = pickle.load(f)
            logging.info(f"Loaded {len(molecule_cache)} items from global cache: {cache_file_path}")
        except Exception as e:
            logging.error(f"Error loading global cache: {e}. Starting with an empty cache.")
            molecule_cache = {}
    else:
        logging.info("Global cache not found. Starting with an empty cache.")
        molecule_cache = {}

def save_global_cache():
    global molecule_cache
    try:
        with open(cache_file_path, "wb") as f:
            pickle.dump(molecule_cache, f)
        logging.info(f"Saved {len(molecule_cache)} items to global cache: {cache_file_path}")
    except Exception as e:
        logging.error(f"Error saving global cache: {e}")

def update_cache_for_smiles_list(smiles_list, n_jobs):
    global molecule_cache
    new_smiles = sorted(list(set(s for s in smiles_list if s and pd.notna(s) and s not in molecule_cache)))
    
    if not new_smiles:
        logging.info("No new SMILES to process for cache update.")
        return

    logging.info(f"Processing {len(new_smiles)} new unique SMILES for cache update...")
    
    batch_size = 5000 # Process in batches to manage memory and save cache periodically
    for i in range(0, len(new_smiles), batch_size):
        batch_smiles = new_smiles[i:i+batch_size]
        logging.info(f"Processing batch {i//batch_size + 1} of {len(new_smiles)//batch_size + 1} for cache...")
        
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_smiles_for_cache)(s) for s in tqdm(batch_smiles, desc="Calculating descriptors/fps", leave=False)
        )
        for s, res_dict in results:
            if s: # Ensure SMILES string is not None
                molecule_cache[s] = res_dict
        
        save_global_cache() # Save after each batch
        gc.collect()


def process_input_file(input_csv_path, output_dir, representation_type, file_label, n_jobs):
    """
    Processes a single input CSV file to extract features or fingerprints.
    Uses the global molecule cache.
    """
    logging.info(f"Processing file: {input_csv_path} for {representation_type}")
    try:
        df = pd.read_csv(input_csv_path, low_memory=False)
    except FileNotFoundError:
        logging.error(f"Input file not found: {input_csv_path}")
        return
    except Exception as e:
        logging.error(f"Error reading input file {input_csv_path}: {e}")
        return
        
    if 'SMILES' not in df.columns:
        logging.error(f"'SMILES' column not found in {input_csv_path}. Cannot process.")
        return
    
    # Ensure cache is up-to-date with all SMILES in the current file
    unique_smiles_in_file = df["SMILES"].dropna().unique().tolist()
    update_cache_for_smiles_list(unique_smiles_in_file, n_jobs)

    output_data_list = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Applying cache to {os.path.basename(input_csv_path)}"):
        smiles = row.get("SMILES")
        # Prepare a base dictionary from the row, excluding mol objects if they were somehow in input
        base_row_dict = {k: v for k, v in row.to_dict().items() if not isinstance(v, Chem.Mol)}

        if smiles and pd.notna(smiles) and smiles in molecule_cache:
            cached_data = molecule_cache[smiles]
            if representation_type == "features":
                # Ensure all target features are present, fill with NaN if missing from cache (should not happen with init)
                features_from_cache = cached_data.get("features", {})
                row_features = {feat_name: features_from_cache.get(feat_name, np.nan) for feat_name in RDKIT_FEATURES_LIST_TARGET}
                base_row_dict.update(row_features)
            elif representation_type == "fingerprints":
                fp_array = cached_data.get("ecfp4")
                if fp_array is not None:
                    # Convert fingerprint array to dictionary of bit columns: 'fp_0', 'fp_1', ...
                    fp_dict = {f"fp_{i}": bit for i, bit in enumerate(fp_array)}
                    base_row_dict.update(fp_dict)
                else: # Fingerprint could not be calculated
                    fp_dict = {f"fp_{i}": np.nan for i in range(2048)} # Fill with NaNs
                    base_row_dict.update(fp_dict)
            output_data_list.append(base_row_dict)
        else: # SMILES is None, NaN, or not in cache (shouldn't happen if update_cache was successful)
            if representation_type == "features":
                row_features = {feat_name: np.nan for feat_name in RDKIT_FEATURES_LIST_TARGET}
                base_row_dict.update(row_features)
            elif representation_type == "fingerprints":
                fp_dict = {f"fp_{i}": np.nan for i in range(2048)}
                base_row_dict.update(fp_dict)
            output_data_list.append(base_row_dict)
            if smiles and pd.notna(smiles):
                 logging.warning(f"SMILES '{smiles}' processed but not found in cache during application. Results will be NaN.")


    output_df = pd.DataFrame(output_data_list)
    
    # Clean up columns - remove mol objects if they were accidentally added from cache to df
    cols_to_drop = [col for col in output_df.columns if isinstance(output_df[col].iloc[0] if len(output_df)>0 else None, Chem.Mol)]
    if cols_to_drop:
        output_df.drop(columns=cols_to_drop, inplace=True)

    output_filename = f"{file_label}_{representation_type}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    try:
        output_df.to_csv(output_path, index=False)
        logging.info(f"Saved {representation_type} to {output_path}")
    except Exception as e:
        logging.error(f"Error saving output file {output_path}: {e}")

    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute molecular features or fingerprints for an input CSV.")
    parser.add_argument("--input_csv", required=True, help="Path to the input CSV file containing a 'SMILES' column.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory to save the output CSV file.")
    parser.add_argument("--representation_type", required=True, choices=["features", "fingerprints"], help="Type of representation to calculate.")
    parser.add_argument("--file_label", required=True, help="A label to include in the output filename (e.g., 'target_X_excluded_chembl').")
    parser.add_argument("--n_jobs", type=int, default=N_JOBS_DEFAULT, help="Number of parallel jobs for descriptor calculation.")
    
    args = parser.parse_args()

    # Load global cache at the beginning of a script run
    load_global_cache()

    process_input_file(args.input_csv, args.output_dir, args.representation_type, args.file_label, args.n_jobs)
    
    # Save global cache at the end (optional, as it's saved after batches too)
    # save_global_cache() # Already saved in batches, final save might be redundant or useful for leftovers
    logging.info("Feature/Fingerprint calculation script finished.")