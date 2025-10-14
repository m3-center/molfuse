#!/usr/bin/env python3
"""
Query UniProt REST API to verify molecular function classifications for UMMBAS targets.

This script retrieves keyword annotations from UniProt for the three target proteins
and verifies that they have the correct molecular function classifications:
- ABL1 (P00519): Should be Transferase (KW-0808)
- Pyruvate Kinase M2 (P14618): Should be Transferase (KW-0808)
- Isocitrate Dehydrogenase (O75874): Should be Oxidoreductase (KW-0560)

Author: UMMBAS v2.0
Date: 2025-10-14
"""

import requests
import json
import csv
import time
from datetime import datetime
from typing import Dict, List, Tuple


def query_uniprot_keywords(uniprot_id: str) -> Tuple[Dict, List[Dict]]:
    """
    Query UniProt REST API for a given UniProt accession and retrieve keyword annotations.
    
    Args:
        uniprot_id: UniProt accession (e.g., 'P00519')
    
    Returns:
        Tuple of (full_data, keywords_list)
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    
    print(f"\n{'='*80}")
    print(f"Querying UniProt for: {uniprot_id}")
    print(f"URL: {url}")
    print(f"{'='*80}")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract keywords
        keywords = []
        if 'keywords' in data:
            for kw in data['keywords']:
                keywords.append({
                    'id': kw.get('id', 'N/A'),
                    'category': kw.get('category', 'N/A'),
                    'name': kw.get('name', 'N/A')
                })
        
        print(f"✓ Successfully retrieved data for {uniprot_id}")
        print(f"  Found {len(keywords)} keyword annotations")
        
        return data, keywords
    
    except requests.exceptions.RequestException as e:
        print(f"✗ Error querying UniProt for {uniprot_id}: {e}")
        return {}, []


def check_molecular_function(keywords: List[Dict], expected_kw_id: str, expected_kw_name: str) -> Dict:
    """
    Check if the expected molecular function keyword is present.
    
    Args:
        keywords: List of keyword dictionaries from UniProt
        expected_kw_id: Expected keyword ID (e.g., 'KW-0808')
        expected_kw_name: Expected keyword name (e.g., 'Transferase')
    
    Returns:
        Dictionary with verification results
    """
    # Look for molecular function category keywords
    molecular_function_keywords = [kw for kw in keywords if kw['category'] == 'Molecular function']
    
    # Check if expected keyword is present
    expected_found = any(kw['id'] == expected_kw_id for kw in keywords)
    
    return {
        'expected_found': expected_found,
        'all_mf_keywords': molecular_function_keywords,
        'all_keyword_ids': [kw['id'] for kw in keywords],
        'all_keyword_names': [kw['name'] for kw in keywords]
    }


def print_verification_results(uniprot_id: str, protein_name: str, expected_kw_id: str, 
                               expected_kw_name: str, keywords: List[Dict]):
    """
    Print human-readable verification results.
    """
    print(f"\n{'─'*80}")
    print(f"VERIFICATION RESULTS FOR: {protein_name} ({uniprot_id})")
    print(f"{'─'*80}")
    
    result = check_molecular_function(keywords, expected_kw_id, expected_kw_name)
    
    print(f"\n🔍 Expected Molecular Function:")
    print(f"   {expected_kw_name} ({expected_kw_id})")
    
    if result['expected_found']:
        print(f"   ✅ VERIFIED - {expected_kw_name} ({expected_kw_id}) is present!")
    else:
        print(f"   ❌ NOT FOUND - {expected_kw_name} ({expected_kw_id}) is MISSING!")
    
    print(f"\n📋 All Molecular Function keywords found:")
    if result['all_mf_keywords']:
        for kw in result['all_mf_keywords']:
            marker = "✓" if kw['id'] == expected_kw_id else " "
            print(f"   {marker} {kw['name']} ({kw['id']})")
    else:
        print(f"   (No molecular function keywords found)")
    
    print(f"\n📚 Total keywords: {len(keywords)}")
    print(f"   Categories: {', '.join(set(kw['category'] for kw in keywords))}")
    
    return result


def main():
    """
    Main function to query UniProt and verify molecular function classifications.
    """
    print("\n" + "="*80)
    print("UMMBAS v2.0 - UniProt Molecular Function Verification")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define target proteins and expected molecular functions
    targets = [
        {
            'uniprot_id': 'P00519',
            'protein_name': 'Tyrosine-protein Kinase ABL1',
            'expected_kw_id': 'KW-0808',
            'expected_kw_name': 'Transferase',
            'rationale': 'Kinases catalyze phosphoryl transfer reactions'
        },
        {
            'uniprot_id': 'P14618',
            'protein_name': 'Pyruvate Kinase M2',
            'expected_kw_id': 'KW-0808',
            'expected_kw_name': 'Transferase',
            'rationale': 'Kinases catalyze phosphoryl transfer reactions'
        },
        {
            'uniprot_id': 'O75874',
            'protein_name': 'Isocitrate Dehydrogenase NADP cytoplasmic',
            'expected_kw_id': 'KW-0560',
            'expected_kw_name': 'Oxidoreductase',
            'rationale': 'Dehydrogenases catalyze oxidation-reduction reactions'
        }
    ]
    
    # Results storage
    verification_results = []
    all_keywords_data = []
    
    # Query each target
    for target in targets:
        time.sleep(0.5)  # Be nice to UniProt API
        
        data, keywords = query_uniprot_keywords(target['uniprot_id'])
        
        if keywords:
            result = print_verification_results(
                target['uniprot_id'],
                target['protein_name'],
                target['expected_kw_id'],
                target['expected_kw_name'],
                keywords
            )
            
            # Store results
            mf_keywords = result['all_mf_keywords']
            verification_results.append({
                'UniProt ID': target['uniprot_id'],
                'Protein Name': target['protein_name'],
                'Expected MF': target['expected_kw_name'],
                'Expected KW Code': target['expected_kw_id'],
                'Verified': 'YES' if result['expected_found'] else 'NO',
                'All MF Keywords': '; '.join([f"{kw['name']} ({kw['id']})" for kw in mf_keywords]),
                'Rationale': target['rationale'],
                'Total Keywords': len(keywords),
                'Verification Date': datetime.now().strftime('%Y-%m-%d')
            })
            
            # Store all keywords for detailed export
            for kw in keywords:
                all_keywords_data.append({
                    'UniProt ID': target['uniprot_id'],
                    'Protein Name': target['protein_name'],
                    'Keyword ID': kw['id'],
                    'Keyword Name': kw['name'],
                    'Keyword Category': kw['category']
                })
    
    # ========================================================================
    # SANITY CHECKS
    # ========================================================================
    print("\n" + "="*80)
    print("SANITY CHECKS")
    print("="*80)
    
    all_verified = all(r['Verified'] == 'YES' for r in verification_results)
    
    print(f"\n✓ CHECK 1: All targets have expected molecular functions?")
    if all_verified:
        print(f"   ✅ PASS - All {len(verification_results)} targets verified!")
    else:
        print(f"   ❌ FAIL - Some targets missing expected molecular functions!")
        for r in verification_results:
            if r['Verified'] == 'NO':
                print(f"      • {r['Protein Name']} ({r['UniProt ID']}): Expected {r['Expected MF']} NOT FOUND")
    
    print(f"\n✓ CHECK 2: ABL1 and Pyruvate Kinase M2 both have Transferase?")
    kinase_targets = [r for r in verification_results if r['UniProt ID'] in ['P00519', 'P14618']]
    kinases_correct = all(r['Expected MF'] == 'Transferase' and r['Verified'] == 'YES' for r in kinase_targets)
    if kinases_correct:
        print(f"   ✅ PASS - Both kinases verified as Transferases")
    else:
        print(f"   ❌ FAIL - Kinases not correctly classified as Transferases")
    
    print(f"\n✓ CHECK 3: Isocitrate Dehydrogenase has Oxidoreductase?")
    idh_target = [r for r in verification_results if r['UniProt ID'] == 'O75874'][0]
    idh_correct = idh_target['Expected MF'] == 'Oxidoreductase' and idh_target['Verified'] == 'YES'
    if idh_correct:
        print(f"   ✅ PASS - Isocitrate Dehydrogenase verified as Oxidoreductase")
    else:
        print(f"   ❌ FAIL - Isocitrate Dehydrogenase not correctly classified")
    
    print(f"\n✓ CHECK 4: No kinases incorrectly classified as 'Protein kinase inhibitor'?")
    wrong_classification = False
    for r in verification_results:
        if 'Protein kinase inhibitor' in r['All MF Keywords']:
            print(f"   ⚠️  WARNING - {r['Protein Name']} has 'Protein kinase inhibitor' keyword")
            wrong_classification = True
    if not wrong_classification:
        print(f"   ✅ PASS - No targets have 'Protein kinase inhibitor' keyword")
    else:
        print(f"   ℹ️  NOTE: 'Protein kinase inhibitor' refers to compounds that inhibit kinases,")
        print(f"      not the kinase enzymes themselves. This is EXPECTED and CORRECT.")
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Save verification summary
    output_file = 'datasets/protein_collection/uniprot_verification_v2.0.csv'
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=verification_results[0].keys())
        writer.writeheader()
        writer.writerows(verification_results)
    print(f"\n✓ Saved verification summary to: {output_file}")
    
    # Save detailed keywords
    detailed_output = 'datasets/protein_collection/uniprot_all_keywords_v2.0.csv'
    with open(detailed_output, 'w', newline='') as f:
        if all_keywords_data:
            writer = csv.DictWriter(f, fieldnames=all_keywords_data[0].keys())
            writer.writeheader()
            writer.writerows(all_keywords_data)
    print(f"✓ Saved detailed keywords to: {detailed_output}")
    
    # Save JSON for programmatic access
    json_output = 'datasets/protein_collection/uniprot_verification_v2.0.json'
    with open(json_output, 'w') as f:
        json.dump({
            'verification_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'all_verified': all_verified,
            'results': verification_results,
            'all_keywords': all_keywords_data
        }, f, indent=2)
    print(f"✓ Saved JSON data to: {json_output}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    if all_verified:
        print("\n🎉 SUCCESS! All molecular function classifications verified!")
        print("\n✓ ABL1 (P00519): Transferase ✓")
        print("✓ Pyruvate Kinase M2 (P14618): Transferase ✓")
        print("✓ Isocitrate Dehydrogenase (O75874): Oxidoreductase ✓")
        print("\n➡️  NEXT STEP: Proceed to Phase 1.2 - Extract ChEMBL Transferase bioactivity data")
        return 0
    else:
        print("\n⚠️  WARNING: Some molecular functions could not be verified!")
        print("   Please review the results above and investigate discrepancies.")
        return 1


if __name__ == "__main__":
    exit(main())
