#!/usr/bin/env python3
"""
Check status of hyperparameter sweep experiments.

Analyzes workspace to identify:
- Completed experiments
- Failed experiments with error details
- Missing/incomplete experiments

Usage:
    python check_hyperparam_status.py [--workspace WORKSPACE_DIR]
"""

import os
import sys
import json
import glob
import argparse
from collections import defaultdict
from datetime import datetime

def extract_config_info(run_dir):
    """Extract configuration info from run directory name or config file."""
    run_name = os.path.basename(run_dir)
    
    # Try to parse from directory name
    parts = run_name.split('_')
    info = {
        'run_dir': run_name,
        'seed': None,
        'representation': None,
        'dr_method': None,
        'n_neighbors': None,
        'min_dist': None
    }
    
    # Extract seed
    for part in parts:
        if part.startswith('seed'):
            info['seed'] = part.replace('seed', '')
    
    # Extract representation
    if 'features' in run_name:
        info['representation'] = 'features'
    elif 'fingerprints' in run_name:
        info['representation'] = 'fingerprints'
    
    # Extract DR method
    if 'pca' in run_name.lower():
        info['dr_method'] = 'PCA'
    elif 'umap' in run_name.lower():
        if 'euclidean' in run_name:
            info['dr_method'] = 'UMAP-Euclidean'
        elif 'jaccard' in run_name:
            info['dr_method'] = 'UMAP-Jaccard'
        else:
            info['dr_method'] = 'UMAP'
    
    # Extract hyperparameters from directory name
    for i, part in enumerate(parts):
        if part.startswith('nn') and len(part) > 2:
            info['n_neighbors'] = part[2:]
        elif part.startswith('md') and len(part) > 2:
            info['min_dist'] = part[2:]
    
    # Try to load from config file
    config_path = os.path.join(run_dir, 'run_config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                dr_methods = config.get('dimensionality_reduction_methods', {})
                if dr_methods:
                    method_key = list(dr_methods.keys())[0]
                    method_config = dr_methods[method_key]
                    info['dr_method'] = method_config.get('short_name', info['dr_method'])
                    info['n_neighbors'] = method_config.get('n_neighbors', info['n_neighbors'])
                    info['min_dist'] = method_config.get('min_dist', info['min_dist'])
        except:
            pass
    
    return info

def check_experiment_status(run_dir, workspace_root, debug_log=None):
    """Check if experiment completed successfully and extract error if failed."""
    status = {
        'completed': False,
        'has_results': False,
        'has_rankings': False,
        'has_metrics': False,
        'error': None,
        'error_type': None,
        'log_file': None
    }
    
    # Extract run name from directory
    run_name = os.path.basename(run_dir)
    
    # Check for orchestrator log in workspace root
    # Pattern: orchestrator_run_*_seed*_config*.log
    # The log name has timestamp between "run_" and "seed*", so we need to match flexibly
    # run_seed42_config_features_pca_projection -> orchestrator_run_*_seed42_config_features_pca_projection.log
    run_name_parts = run_name.replace('run_', '', 1)  # Remove 'run_' prefix
    search_pattern = os.path.join(workspace_root, f'orchestrator_run_*_{run_name_parts}.log')
    
    if debug_log:
        debug_log.write(f"\n{'='*80}\n")
        debug_log.write(f"Run directory: {run_name}\n")
        debug_log.write(f"Search pattern: {search_pattern}\n")
    
    orchestrator_logs = glob.glob(search_pattern)
    
    if debug_log:
        debug_log.write(f"Found {len(orchestrator_logs)} log files: {orchestrator_logs}\n")
    
    if orchestrator_logs:
        log_file = orchestrator_logs[0]
        status['log_file'] = os.path.basename(log_file)
        
        if debug_log:
            debug_log.write(f"Reading log file: {log_file}\n")
        
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
                
                # Check for completion
                if 'STEP 7: Script finished' in log_content or 'Orchestration complete' in log_content:
                    status['completed'] = True
                    if debug_log:
                        debug_log.write(f"✅ Completion marker found\n")
                
                # Check for errors
                if 'ERROR' in log_content or 'Error' in log_content or 'Traceback' in log_content:
                    # Extract error information
                    lines = log_content.split('\n')
                    error_lines = []
                    in_traceback = False
                    
                    for line in lines:
                        if 'ERROR' in line or 'Error' in line:
                            error_lines.append(line.strip())
                        elif 'Traceback' in line:
                            in_traceback = True
                            error_lines.append(line.strip())
                        elif in_traceback:
                            error_lines.append(line.strip())
                            if line.strip() and not line.startswith(' '):
                                in_traceback = False
                    
                    if error_lines:
                        status['error'] = '\n'.join(error_lines[-10:])  # Last 10 lines
                        
                        # Categorize error type
                        error_text = ' '.join(error_lines).lower()
                        if 'filenotfounderror' in error_text or 'no such file' in error_text:
                            if 'model' in error_text:
                                status['error_type'] = 'Missing Model File'
                            elif 'scaler' in error_text:
                                status['error_type'] = 'Missing Scaler File'
                            else:
                                status['error_type'] = 'File Not Found'
                        elif 'memoryerror' in error_text or 'out of memory' in error_text:
                            status['error_type'] = 'Out of Memory'
                        elif 'valueerror' in error_text:
                            status['error_type'] = 'Value Error'
                        elif 'keyerror' in error_text:
                            status['error_type'] = 'Key Error'
                        elif 'cuda' in error_text or 'gpu' in error_text:
                            status['error_type'] = 'GPU/CUDA Error'
                        else:
                            status['error_type'] = 'Other Error'
                        
                        if debug_log:
                            debug_log.write(f"❌ Error detected: {status['error_type']}\n")
        except Exception as e:
            if debug_log:
                debug_log.write(f"⚠️  Error reading log file: {e}\n")
    else:
        if debug_log:
            debug_log.write(f"⚠️  No orchestrator log found\n")
    
    # Check for results files
    results_pattern = os.path.join(run_dir, '*/results/*/*/dim_*/*')
    results_dirs = glob.glob(results_pattern)
    
    if debug_log:
        debug_log.write(f"Results pattern: {results_pattern}\n")
        debug_log.write(f"Found {len(results_dirs)} result directories\n")
    
    if results_dirs:
        status['has_results'] = True
        
        # Check for ranking files
        for results_dir in results_dirs:
            ranking_files = glob.glob(os.path.join(results_dir, '*-RANKED.csv'))
            if ranking_files:
                status['has_rankings'] = True
                break
        
        # Check for metrics files
        for results_dir in results_dirs:
            metrics_files = glob.glob(os.path.join(results_dir, '*_ranking_metrics.csv'))
            if metrics_files:
                status['has_metrics'] = True
                break
    
    # If we have ranking metrics but no log completion marker, consider it completed
    # (The analysis phase completed successfully even if log wasn't captured)
    if status['has_metrics'] and not status['completed']:
        status['completed'] = True
        if debug_log:
            debug_log.write(f"✅ Marked complete based on metrics file presence\n")
    
    if debug_log:
        debug_log.write(f"Final status: completed={status['completed']}, "
                       f"has_results={status['has_results']}, "
                       f"has_rankings={status['has_rankings']}, "
                       f"has_metrics={status['has_metrics']}\n")
    
    return status

def main():
    parser = argparse.ArgumentParser(description='Check hyperparameter sweep status')
    parser.add_argument('--workspace', default='experiment_workspace_hyperparam_sweep_v2',
                       help='Workspace directory to check')
    args = parser.parse_args()
    
    workspace = args.workspace
    
    if not os.path.exists(workspace):
        print(f"ERROR: Workspace directory not found: {workspace}")
        sys.exit(1)
    
    print("=" * 80)
    print(f"HYPERPARAMETER SWEEP STATUS CHECK")
    print(f"Workspace: {workspace}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # Find all run directories
    run_dirs = glob.glob(os.path.join(workspace, 'run_seed*'))
    
    if not run_dirs:
        print("No run directories found!")
        sys.exit(1)
    
    print(f"Found {len(run_dirs)} run directories")
    print()
    
    # Open debug log
    debug_log_path = os.path.join(workspace, f'status_check_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    print(f"Writing debug log to: {debug_log_path}")
    print()
    
    with open(debug_log_path, 'w') as debug_log:
        debug_log.write(f"Status Check Debug Log\n")
        debug_log.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        debug_log.write(f"Workspace: {workspace}\n")
        debug_log.write(f"Total run directories: {len(run_dirs)}\n")
        
        # Analyze each run
        results = []
        for run_dir in sorted(run_dirs):
            config_info = extract_config_info(run_dir)
            status = check_experiment_status(run_dir, workspace, debug_log)
            
            results.append({
                **config_info,
                **status
            })
    
        # Summary statistics
        completed = [r for r in results if r['completed']]
        failed = [r for r in results if not r['completed'] and r['error']]
        incomplete = [r for r in results if not r['completed'] and not r['error']]
        
        debug_log.write(f"\n{'='*80}\n")
        debug_log.write(f"SUMMARY\n")
        debug_log.write(f"Completed: {len(completed)}\n")
        debug_log.write(f"Failed: {len(failed)}\n")
        debug_log.write(f"Incomplete: {len(incomplete)}\n")
    
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total experiments:      {len(results)}")
    print(f"✅ Completed:           {len(completed)} ({len(completed)/len(results)*100:.1f}%)")
    print(f"❌ Failed with error:   {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
    print(f"⏳ Incomplete/Running:  {len(incomplete)} ({len(incomplete)/len(results)*100:.1f}%)")
    print()
    
    # Group by representation and method
    by_repr_method = defaultdict(lambda: {'completed': 0, 'failed': 0, 'incomplete': 0})
    for r in results:
        key = f"{r['representation']} - {r['dr_method']}"
        if r['completed']:
            by_repr_method[key]['completed'] += 1
        elif r['error']:
            by_repr_method[key]['failed'] += 1
        else:
            by_repr_method[key]['incomplete'] += 1
    
    print("=" * 80)
    print("BY METHOD")
    print("=" * 80)
    print(f"{'Method':<30} {'Completed':>10} {'Failed':>10} {'Incomplete':>10}")
    print("-" * 80)
    for key in sorted(by_repr_method.keys()):
        stats = by_repr_method[key]
        print(f"{key:<30} {stats['completed']:>10} {stats['failed']:>10} {stats['incomplete']:>10}")
    print()
    
    # Error types
    if failed:
        print("=" * 80)
        print("ERROR TYPES")
        print("=" * 80)
        error_counts = defaultdict(int)
        for r in failed:
            error_counts[r['error_type']] += 1
        
        for error_type, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"{error_type:<30} {count:>5} occurrences")
        print()
    
    # Detailed failed experiments
    if failed:
        print("=" * 80)
        print("FAILED EXPERIMENTS DETAILS")
        print("=" * 80)
        
        # Group by error type
        by_error_type = defaultdict(list)
        for r in failed:
            by_error_type[r['error_type']].append(r)
        
        for error_type, experiments in sorted(by_error_type.items()):
            print(f"\n{'='*80}")
            print(f"Error Type: {error_type} ({len(experiments)} experiments)")
            print(f"{'='*80}")
            
            for exp in experiments[:5]:  # Show first 5 of each type
                print(f"\nRun: {exp['run_dir']}")
                print(f"  Representation: {exp['representation']}")
                print(f"  Method: {exp['dr_method']}")
                if exp['n_neighbors']:
                    print(f"  n_neighbors: {exp['n_neighbors']}")
                if exp['min_dist']:
                    print(f"  min_dist: {exp['min_dist']}")
                print(f"  Seed: {exp['seed']}")
                if exp['error']:
                    print(f"  Error (last 3 lines):")
                    error_lines = exp['error'].split('\n')
                    for line in error_lines[-3:]:
                        if line.strip():
                            print(f"    {line}")
            
            if len(experiments) > 5:
                print(f"\n  ... and {len(experiments) - 5} more with same error type")
    
    # Export to CSV
    output_file = os.path.join(workspace, f'status_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    try:
        import csv
        with open(output_file, 'w', newline='') as f:
            fieldnames = ['run_dir', 'seed', 'representation', 'dr_method', 'n_neighbors', 'min_dist',
                         'completed', 'has_results', 'has_rankings', 'has_metrics', 'error_type', 'log_file']
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)
        print()
        print("=" * 80)
        print(f"✅ Detailed report saved to: {output_file}")
        print(f"✅ Debug log saved to: {debug_log_path}")
        print("=" * 80)
    except Exception as e:
        print(f"\n⚠️  Could not save CSV report: {e}")

if __name__ == '__main__':
    main()
