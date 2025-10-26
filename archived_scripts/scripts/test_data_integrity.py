#!/usr/bin/env python3
"""
Test Data Integrity: Validate Non-Contaminated Datasets

This script verifies that prepare_data.py successfully created clean datasets with:
1. No overlap between target ligands and MF cloud
2. No overlap between target ligands and ZINC decoys
3. No overlap between MF cloud and ZINC decoys

Usage:
    python test_data_integrity.py --config experiment_config.json --target_id_name <target>
    python test_data_integrity.py --config experiment_config.json --target_id_name TyrosinaseTetramer_Q02772 --prepared_dir datasets/TyrosinaseTetramer_Q02772
"""

import pandas as pd
import os
import argparse
import json
import logging
from typing import Set, Tuple, Dict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_data_integrity.log"),
        logging.StreamHandler()
    ]
)

class DataIntegrityTester:
    """Test suite for validating dataset non-contamination"""
    
    def __init__(self, prepared_data_dir: str, target_id_name: str):
        self.prepared_data_dir = prepared_data_dir
        self.target_id_name = target_id_name
        self.results = {
            'passed': [],
            'failed': [],
            'warnings': []
        }
    
    def load_smiles_from_csv(self, filepath: str, label: str) -> Set[str]:
        """Load unique SMILES from a CSV file"""
        if not os.path.exists(filepath):
            logging.warning(f"File not found: {filepath}")
            self.results['warnings'].append(f"Missing file: {filepath}")
            return set()
        
        try:
            df = pd.read_csv(filepath, low_memory=False)
            if 'SMILES' not in df.columns:
                logging.error(f"'SMILES' column not found in {filepath}")
                self.results['failed'].append(f"Missing SMILES column in {label}")
                return set()
            
            smiles_set = set(df['SMILES'].dropna().unique())
            logging.info(f"Loaded {len(smiles_set)} unique SMILES from {label} ({os.path.basename(filepath)})")
            return smiles_set
        
        except Exception as e:
            logging.error(f"Error loading {filepath}: {e}")
            self.results['failed'].append(f"Error loading {label}: {e}")
            return set()
    
    def test_no_overlap(self, set1: Set[str], set2: Set[str], 
                        label1: str, label2: str) -> bool:
        """Test that two SMILES sets have no overlap"""
        overlap = set1.intersection(set2)
        
        if len(overlap) == 0:
            msg = f"✅ PASS: No overlap between {label1} and {label2}"
            logging.info(msg)
            self.results['passed'].append(msg)
            return True
        else:
            msg = f"❌ FAIL: Found {len(overlap)} overlapping SMILES between {label1} and {label2}"
            logging.error(msg)
            self.results['failed'].append(msg)
            
            # Log first 5 overlapping SMILES for debugging
            for i, smiles in enumerate(list(overlap)[:5], 1):
                logging.error(f"  Overlap {i}: {smiles}")
            if len(overlap) > 5:
                logging.error(f"  ... and {len(overlap) - 5} more")
            
            return False
    
    def test_target_ligands_exist(self, target_smiles: Set[str]) -> bool:
        """Test that we have target ligands (sanity check)"""
        if len(target_smiles) == 0:
            msg = "⚠️  WARNING: No target ligands found (may be expected for some proteins)"
            logging.warning(msg)
            self.results['warnings'].append(msg)
            return True  # Not a failure, just a warning
        else:
            msg = f"✅ PASS: Found {len(target_smiles)} target ligand SMILES"
            logging.info(msg)
            self.results['passed'].append(msg)
            return True
    
    def test_datasets_exist(self) -> bool:
        """Test that all expected files exist"""
        expected_files = [
            f"{self.target_id_name}_target_ligands_for_feature_calc_raw.csv",
            f"{self.target_id_name}_chembl_mf_excluded_features.csv",
            f"{self.target_id_name}_chembl_mf_excluded_fingerprints.csv",
            f"{self.target_id_name}_zinc_excluded_features.csv",
            f"{self.target_id_name}_zinc_excluded_fingerprints.csv"
        ]
        
        all_exist = True
        for filename in expected_files:
            filepath = os.path.join(self.prepared_data_dir, filename)
            if os.path.exists(filepath):
                msg = f"✅ File exists: {filename}"
                logging.info(msg)
            else:
                msg = f"❌ FAIL: Missing file: {filename}"
                logging.error(msg)
                self.results['failed'].append(msg)
                all_exist = False
        
        if all_exist:
            self.results['passed'].append("All expected dataset files exist")
        
        return all_exist
    
    def test_dataset_sizes(self, target_smiles: Set[str], mf_features: Set[str], 
                          mf_fingerprints: Set[str], zinc_features: Set[str], 
                          zinc_fingerprints: Set[str]) -> bool:
        """Test that dataset sizes are reasonable"""
        
        # Check MF features vs fingerprints consistency
        if len(mf_features) > 0 and len(mf_fingerprints) > 0:
            if mf_features == mf_fingerprints:
                msg = f"✅ PASS: MF features and fingerprints have identical SMILES ({len(mf_features)} molecules)"
                logging.info(msg)
                self.results['passed'].append(msg)
            else:
                msg = f"⚠️  WARNING: MF features ({len(mf_features)}) and fingerprints ({len(mf_fingerprints)}) have different SMILES counts"
                logging.warning(msg)
                self.results['warnings'].append(msg)
        
        # Check ZINC features vs fingerprints consistency
        if len(zinc_features) > 0 and len(zinc_fingerprints) > 0:
            if zinc_features == zinc_fingerprints:
                msg = f"✅ PASS: ZINC features and fingerprints have identical SMILES ({len(zinc_features)} molecules)"
                logging.info(msg)
                self.results['passed'].append(msg)
            else:
                msg = f"⚠️  WARNING: ZINC features ({len(zinc_features)}) and fingerprints ({len(zinc_fingerprints)}) have different SMILES counts"
                logging.warning(msg)
                self.results['warnings'].append(msg)
        
        # Check that ZINC is substantially larger than MF (sanity check)
        if len(zinc_features) > 0 and len(mf_features) > 0:
            ratio = len(zinc_features) / len(mf_features)
            if ratio > 0.5:  # ZINC should be at least somewhat comparable to MF
                msg = f"✅ PASS: ZINC ({len(zinc_features)}) vs MF ({len(mf_features)}) ratio: {ratio:.2f}x"
                logging.info(msg)
                self.results['passed'].append(msg)
            else:
                msg = f"⚠️  WARNING: ZINC ({len(zinc_features)}) is much smaller than MF ({len(mf_features)}). Ratio: {ratio:.2f}x"
                logging.warning(msg)
                self.results['warnings'].append(msg)
        
        return True
    
    def run_all_tests(self) -> Dict:
        """Run complete test suite"""
        logging.info("="*80)
        logging.info(f"DATA INTEGRITY TEST SUITE: {self.target_id_name}")
        logging.info("="*80)
        
        # Test 1: Check file existence
        logging.info("\nTest 1: Checking dataset file existence...")
        self.test_datasets_exist()
        
        # Load all datasets
        logging.info("\nLoading datasets...")
        target_smiles = self.load_smiles_from_csv(
            os.path.join(self.prepared_data_dir, f"{self.target_id_name}_target_ligands_for_feature_calc_raw.csv"),
            "Target Ligands"
        )
        
        mf_features_smiles = self.load_smiles_from_csv(
            os.path.join(self.prepared_data_dir, f"{self.target_id_name}_chembl_mf_excluded_features.csv"),
            "MF Cloud (features)"
        )
        
        mf_fingerprints_smiles = self.load_smiles_from_csv(
            os.path.join(self.prepared_data_dir, f"{self.target_id_name}_chembl_mf_excluded_fingerprints.csv"),
            "MF Cloud (fingerprints)"
        )
        
        zinc_features_smiles = self.load_smiles_from_csv(
            os.path.join(self.prepared_data_dir, f"{self.target_id_name}_zinc_excluded_features.csv"),
            "ZINC Decoys (features)"
        )
        
        zinc_fingerprints_smiles = self.load_smiles_from_csv(
            os.path.join(self.prepared_data_dir, f"{self.target_id_name}_zinc_excluded_fingerprints.csv"),
            "ZINC Decoys (fingerprints)"
        )
        
        # Test 2: Target ligands sanity check
        logging.info("\nTest 2: Checking target ligands exist...")
        self.test_target_ligands_exist(target_smiles)
        
        # Test 3: Dataset size consistency
        logging.info("\nTest 3: Checking dataset size consistency...")
        self.test_dataset_sizes(target_smiles, mf_features_smiles, mf_fingerprints_smiles,
                               zinc_features_smiles, zinc_fingerprints_smiles)
        
        # Test 4-9: No overlap tests (the critical contamination checks)
        logging.info("\nTest 4: Checking Target vs MF Cloud (features) - NO OVERLAP EXPECTED...")
        self.test_no_overlap(target_smiles, mf_features_smiles, "Target Ligands", "MF Cloud (features)")
        
        logging.info("\nTest 5: Checking Target vs MF Cloud (fingerprints) - NO OVERLAP EXPECTED...")
        self.test_no_overlap(target_smiles, mf_fingerprints_smiles, "Target Ligands", "MF Cloud (fingerprints)")
        
        logging.info("\nTest 6: Checking Target vs ZINC (features) - NO OVERLAP EXPECTED...")
        self.test_no_overlap(target_smiles, zinc_features_smiles, "Target Ligands", "ZINC Decoys (features)")
        
        logging.info("\nTest 7: Checking Target vs ZINC (fingerprints) - NO OVERLAP EXPECTED...")
        self.test_no_overlap(target_smiles, zinc_fingerprints_smiles, "Target Ligands", "ZINC Decoys (fingerprints)")
        
        logging.info("\nTest 8: Checking MF Cloud vs ZINC (features) - NO OVERLAP EXPECTED...")
        self.test_no_overlap(mf_features_smiles, zinc_features_smiles, "MF Cloud (features)", "ZINC Decoys (features)")
        
        logging.info("\nTest 9: Checking MF Cloud vs ZINC (fingerprints) - NO OVERLAP EXPECTED...")
        self.test_no_overlap(mf_fingerprints_smiles, zinc_fingerprints_smiles, "MF Cloud (fingerprints)", "ZINC Decoys (fingerprints)")
        
        # Generate summary report
        self.generate_summary_report()
        
        return self.results
    
    def generate_summary_report(self):
        """Generate and display summary report"""
        logging.info("\n" + "="*80)
        logging.info("TEST SUMMARY REPORT")
        logging.info("="*80)
        
        total_tests = len(self.results['passed']) + len(self.results['failed'])
        passed_count = len(self.results['passed'])
        failed_count = len(self.results['failed'])
        warning_count = len(self.results['warnings'])
        
        logging.info(f"Total Tests Run: {total_tests}")
        logging.info(f"✅ Passed: {passed_count}")
        logging.info(f"❌ Failed: {failed_count}")
        logging.info(f"⚠️  Warnings: {warning_count}")
        
        if failed_count == 0:
            logging.info("\n🎉 ALL TESTS PASSED - DATASETS ARE NON-CONTAMINATED!")
        else:
            logging.error("\n⚠️  SOME TESTS FAILED - DATA CONTAMINATION DETECTED!")
            logging.error("\nFailed tests:")
            for failure in self.results['failed']:
                logging.error(f"  - {failure}")
        
        if warning_count > 0:
            logging.warning("\nWarnings:")
            for warning in self.results['warnings']:
                logging.warning(f"  - {warning}")
        
        logging.info("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Test data integrity of prepared datasets (target ligands, MF cloud, ZINC decoys)"
    )
    parser.add_argument("--config", required=True, help="Path to experiment_config.json")
    parser.add_argument("--target_id_name", required=True, help="Target ID name (e.g., TyrosinaseTetramer_Q02772)")
    parser.add_argument("--prepared_dir", help="Override prepared data directory (default: datasets/{target_id_name})")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Determine prepared data directory
    if args.prepared_dir:
        prepared_dir = args.prepared_dir
    else:
        # Default: datasets/{target_id_name}
        prepared_dir = os.path.join("datasets", args.target_id_name)
    
    if not os.path.exists(prepared_dir):
        logging.error(f"Prepared data directory not found: {prepared_dir}")
        logging.error("Please run prepare_data.py first or specify --prepared_dir")
        return 1
    
    logging.info(f"Testing prepared data in: {prepared_dir}")
    
    # Run test suite
    tester = DataIntegrityTester(prepared_dir, args.target_id_name)
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    if len(results['failed']) == 0:
        return 0  # Success
    else:
        return 1  # Failure


if __name__ == "__main__":
    exit(main())
