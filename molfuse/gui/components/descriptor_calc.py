"""Descriptor calculator: Computes Mordred 2D molecular descriptors from SMILES."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from mordred import Calculator, descriptors as mordred_descriptors_module


logger = logging.getLogger(__name__)


# Mordred descriptor list matching the 40-feature set used in MolFuSE training
MORDRED_DESCRIPTOR_LIST = [
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
]


def calculate_dipole_moment(mol):
    """Calculate dipole moment using Gasteiger charges."""
    if mol is None:
        return np.nan
    try:
        mol_h = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol_h, randomSeed=42) != 0:
            return np.nan
        AllChem.UFFOptimizeMolecule(mol_h)
        AllChem.ComputeGasteigerCharges(mol_h)
        conf = mol_h.GetConformer()
        dipole_vector = np.array([0.0, 0.0, 0.0])
        for atom in mol_h.GetAtoms():
            pos = np.array(conf.GetAtomPosition(atom.GetIdx()))
            charge_str = atom.GetProp('_GasteigerCharge')
            if not np.isfinite(float(charge_str)):
                return np.nan
            dipole_vector += float(charge_str) * pos
        return np.linalg.norm(dipole_vector) * 4.80320427
    except Exception:
        return np.nan


class DescriptorCalculator:
    """Computes Mordred 2D molecular descriptors matching MolFuSE training."""
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        """Initialize with 40 Mordred descriptors + DipoleMoment."""
        self.mordred_calc = Calculator(MORDRED_DESCRIPTOR_LIST, ignore_3D=True)
        self.mordred_descriptor_names = [str(d) for d in self.mordred_calc.descriptors]
        all_descriptor_names = ['DipoleMoment'] + self.mordred_descriptor_names
        
        if feature_names is not None:
            self.feature_names = [d for d in feature_names if d in all_descriptor_names]
        else:
            self.feature_names = all_descriptor_names
        
        logger.info(f"Initialized DescriptorCalculator with {len(self.feature_names)} Mordred descriptors")
    
    def compute_for_smiles(self, smiles: str) -> Dict[str, float]:
        """Compute descriptors for single SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {}
        
        descriptors = {}
        if 'DipoleMoment' in self.feature_names:
            descriptors['DipoleMoment'] = calculate_dipole_moment(mol)
        
        try:
            mordred_result = self.mordred_calc(mol)
            for i, name in enumerate(self.mordred_descriptor_names):
                if name in self.feature_names:
                    value = mordred_result[i]
                    descriptors[name] = float(value) if isinstance(value, (int, float)) and np.isfinite(value) else np.nan
        except Exception:
            for name in self.mordred_descriptor_names:
                if name in self.feature_names:
                    descriptors[name] = np.nan
        
        return descriptors
    
    def compute_for_smiles_list(self, smiles_list: List[str], include_smiles: bool = True) -> pd.DataFrame:
        """Compute descriptors for list of SMILES."""
        logger.info(f"Computing descriptors for {len(smiles_list)} molecules...")
        results = []
        
        for smiles in smiles_list:
            row = {'SMILES': smiles} if include_smiles else {}
            desc_dict = self.compute_for_smiles(smiles)
            if not desc_dict:
                for feat_name in self.feature_names:
                    row[feat_name] = np.nan
            else:
                row.update(desc_dict)
            results.append(row)
        
        df = pd.DataFrame(results)
        logger.info(f"Computed descriptors: shape={df.shape}")
        return df
