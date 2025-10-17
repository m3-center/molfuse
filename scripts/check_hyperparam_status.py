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
import csv
import shutil
import statistics
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm  # Progress bars

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
        'min_dist': None,
        'dim': None  # Add dimensionality extraction
    }
    
    # Extract seed
    for part in parts:
        if part.startswith('seed'):
            info['seed'] = part.replace('seed', '')
    
    # Extract dimensionality (dim2, dim5, dim10, etc.)
    for part in parts:
        if part.startswith('dim') and len(part) > 3:
            info['dim'] = part[3:]  # Extract number after 'dim'
    
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
                    # Extract dimensionality from config if not already found
                    if not info['dim']:
                        simspace_dim = config.get('global_settings', {}).get('simspace_dim')
                        if simspace_dim:
                            info['dim'] = str(simspace_dim)
        except:
            pass
    
    return info

def check_experiment_status(run_dir, workspace_root, debug_log=None):
    """Check if experiment completed successfully and extract error if failed.
    
    Strategy: Check for output files FIRST (ranking metrics = success), 
    then check logs for error details only if needed.
    """
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
    
    if debug_log:
        debug_log.write(f"\n{'='*80}\n")
        debug_log.write(f"Run directory: {run_name}\n")
    
    # ============================================================================
    # STEP 1: Check for output files (PRIMARY success indicator)
    # ============================================================================
    # Pattern: run_seed*/TARGET/results/REPR/dim_N/METHOD/
    # Example: run_seed46.../TyrosineProteinKinaseABL1_P00519/results/features/dim_2/PCA/
    results_pattern = os.path.join(run_dir, '*/results/*/dim_*/*')
    results_dirs = glob.glob(results_pattern)
    
    if debug_log:
        debug_log.write(f"Results pattern: {results_pattern}\n")
        debug_log.write(f"Found {len(results_dirs)} result directories\n")
        if results_dirs:
            debug_log.write(f"Example: {results_dirs[0]}\n")
    
    if results_dirs:
        status['has_results'] = True
        
        # Check for ranking files
        for results_dir in results_dirs:
            ranking_files = glob.glob(os.path.join(results_dir, '*-RANKED.csv'))
            if ranking_files:
                status['has_rankings'] = True
                if debug_log:
                    debug_log.write(f"✅ Found ranking file: {ranking_files[0]}\n")
                break
        
        # Check for metrics files (KEY success indicator)
        for results_dir in results_dirs:
            metrics_files = glob.glob(os.path.join(results_dir, '*_ranking_metrics.csv'))
            if metrics_files:
                status['has_metrics'] = True
                if debug_log:
                    debug_log.write(f"✅ Found metrics file: {metrics_files[0]}\n")
                break
    
    # If we have ranking metrics, the experiment completed successfully
    # (Even if there were warnings/errors like CUDA fallback to CPU)
    if status['has_metrics']:
        status['completed'] = True
        if debug_log:
            debug_log.write(f"✅ Marked COMPLETED based on metrics file presence\n")
    
    # ============================================================================
    # STEP 2: If NOT completed, check logs for error details
    # ============================================================================
    if not status['completed']:
        # Check for orchestrator log in repo root (one level up from workspace)
        # Pattern: orchestrator_run_*_seed*_config*.log
        run_name_parts = run_name.replace('run_', '', 1)  # Remove 'run_' prefix
        repo_root = os.path.dirname(os.path.abspath(workspace_root))  # Go one level up
        search_pattern = os.path.join(repo_root, f'orchestrator_run_*_{run_name_parts}.log')
        
        if debug_log:
            debug_log.write(f"Repo root: {repo_root}\n")
            debug_log.write(f"Log search pattern: {search_pattern}\n")
        
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
                    log_content_lower = log_content.lower()
                    
                    # Check for return code -9 (SIGKILL - usually OOM)
                    if 'return code -9' in log_content_lower or 'failed with return code -9' in log_content_lower:
                        lines = log_content.split('\n')
                        error_lines = []
                        # Find lines with return code -9
                        for i, line in enumerate(lines):
                            if 'return code -9' in line.lower():
                                # Capture context around error (5 lines before, 5 after)
                                start = max(0, i - 5)
                                end = min(len(lines), i + 6)
                                error_lines = lines[start:end]
                                break
                        
                        if error_lines:
                            status['error'] = '\n'.join([l.strip() for l in error_lines if l.strip()])
                            status['error_type'] = 'Process Killed (return code -9, likely OOM)'
                            if debug_log:
                                debug_log.write(f"❌ Process killed (return code -9) detected - likely OOM\n")
                    
                    # Check for explicit OOM errors
                    elif 'oom' in log_content_lower or 'out of memory' in log_content_lower or 'memoryerror' in log_content_lower:
                        lines = log_content.split('\n')
                        error_lines = []
                        # Find lines around OOM error
                        for i, line in enumerate(lines):
                            if 'oom' in line.lower() or 'out of memory' in line.lower() or 'memoryerror' in line.lower():
                                # Capture context around error (5 lines before, 5 after)
                                start = max(0, i - 5)
                                end = min(len(lines), i + 6)
                                error_lines = lines[start:end]
                                break
                        
                        if error_lines:
                            status['error'] = '\n'.join([l.strip() for l in error_lines if l.strip()])
                            status['error_type'] = 'Out of Memory (OOM)'
                            if debug_log:
                                debug_log.write(f"❌ OOM Error detected in log\n")
                    
                    # Check for FAILED messages (generic failure)
                    elif ' FAILED ' in log_content or ' failed ' in log_content_lower:
                        lines = log_content.split('\n')
                        error_lines = []
                        # Find lines with FAILED
                        for i, line in enumerate(lines):
                            if ' failed ' in line.lower():
                                # Capture context around error (3 lines before, 3 after)
                                start = max(0, i - 3)
                                end = min(len(lines), i + 4)
                                error_lines = lines[start:end]
                                break
                        
                        if error_lines:
                            status['error'] = '\n'.join([l.strip() for l in error_lines if l.strip()])
                            status['error_type'] = 'Task Failed'
                            if debug_log:
                                debug_log.write(f"❌ Task failure detected in log\n")
                    
                    # Check for other fatal errors (that stopped execution)
                    elif 'ERROR' in log_content or 'Error' in log_content or 'Traceback' in log_content:
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
                            # Categorize error type
                            error_text = ' '.join(error_lines).lower()
                            
                            # IGNORE CUDA errors - they are warnings, code falls back to CPU
                            if 'cuda' in error_text or 'gpu' in error_text or 'numba.cuda' in error_text:
                                if debug_log:
                                    debug_log.write(f"⚠️  CUDA warning detected but IGNORED (fallback to CPU)\n")
                                # Don't set error status for CUDA warnings
                                pass
                            elif 'filenotfounderror' in error_text or 'no such file' in error_text:
                                status['error'] = '\n'.join(error_lines[-10:])  # Last 10 lines
                                if 'model' in error_text:
                                    status['error_type'] = 'Missing Model File'
                                elif 'scaler' in error_text:
                                    status['error_type'] = 'Missing Scaler File'
                                else:
                                    status['error_type'] = 'File Not Found'
                            elif 'valueerror' in error_text:
                                status['error'] = '\n'.join(error_lines[-10:])
                                status['error_type'] = 'Value Error'
                            elif 'keyerror' in error_text:
                                status['error'] = '\n'.join(error_lines[-10:])
                                status['error_type'] = 'Key Error'
                            else:
                                status['error'] = '\n'.join(error_lines[-10:])
                                status['error_type'] = 'Other Error'
                            
                            if status['error_type'] and debug_log:
                                debug_log.write(f"❌ Error detected in log: {status['error_type']}\n")
            except Exception as e:
                if debug_log:
                    debug_log.write(f"⚠️  Error reading log file: {e}\n")
        else:
            if debug_log:
                debug_log.write(f"⚠️  No orchestrator log found - experiment may be running\n")
    else:
        # Even if completed, record which log file corresponds to this run
        run_name_parts = run_name.replace('run_', '', 1)
        repo_root = os.path.dirname(os.path.abspath(workspace_root))
        search_pattern = os.path.join(repo_root, f'orchestrator_run_*_{run_name_parts}.log')
        orchestrator_logs = glob.glob(search_pattern)
        if orchestrator_logs:
            status['log_file'] = os.path.basename(orchestrator_logs[0])
    
    if debug_log:
        debug_log.write(f"Final status: completed={status['completed']}, "
                       f"has_results={status['has_results']}, "
                       f"has_rankings={status['has_rankings']}, "
                       f"has_metrics={status['has_metrics']}, "
                       f"error_type={status['error_type']}\n")
    
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
        
        # Analyze each run with progress bar
        results = []
        print("Checking experiment status...")
        for run_dir in tqdm(sorted(run_dirs), desc="Analyzing experiments", unit="exp"):
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
    
    # ============================================================================
    # PRELIMINARY RANKING METRICS (EF@1% averaged across seeds)
    # ============================================================================
    if completed:
        print("=" * 80)
        print("PRELIMINARY RANKING METRICS (Completed Experiments Only)")
        print("=" * 80)
        
        # Collect EF@1% scores for each method, organized by dimensionality
        import csv as csv_module
        metrics_by_method_and_dim = defaultdict(lambda: defaultdict(list))
        
        print("\nCollecting metrics from experiment results...")
        for r in tqdm(completed, desc="Reading metrics files", unit="exp"):
            # Find metrics file for this run
            run_dir = os.path.join(workspace, r['run_dir'])
            # Pattern: TARGET/results/REPR/dim_N/METHOD/*_ranking_metrics.csv
            metrics_pattern = os.path.join(run_dir, '*/results/*/dim_*/*/*_ranking_metrics.csv')
            metrics_files = glob.glob(metrics_pattern)
            
            if metrics_files:
                try:
                    with open(metrics_files[0], 'r') as f:
                        reader = csv_module.DictReader(f)
                        for row in reader:
                            # Try different column name variations
                            ef_key = None
                            if 'ef_1%' in row:
                                ef_key = 'ef_1%'
                            elif 'EF@1%' in row:
                                ef_key = 'EF@1%'
                            elif 'EF@1' in row:
                                ef_key = 'EF@1'
                            elif 'ef_1' in row:
                                ef_key = 'ef_1'
                            
                            if ef_key:
                                ef_value = float(row[ef_key])
                                
                                # Get dimensionality
                                dim = r.get('dim', 'unknown')
                                
                                # Create method key with hyperparameters
                                if r['n_neighbors'] and r['min_dist']:
                                    method_key = f"{r['representation']}-{r['dr_method']}-nn{r['n_neighbors']}-md{r['min_dist']}"
                                else:
                                    method_key = f"{r['representation']}-{r['dr_method']}"
                                
                                metrics_by_method_and_dim[dim][method_key].append(ef_value)
                                break
                except Exception as e:
                    pass  # Skip if can't read metrics
        
        print()  # Newline after progress bar
        
        if metrics_by_method_and_dim:
            # Calculate averages and display BY DIMENSIONALITY
            import statistics
            
            # Get all unique dimensions and sort them
            all_dims = sorted(metrics_by_method_and_dim.keys(), 
                            key=lambda x: int(x) if x.isdigit() else 999)
            
            for dim in all_dims:
                print(f"\n{'='*80}")
                print(f"DIMENSIONALITY: {dim}D")
                print(f"{'='*80}")
                
                metrics_by_method = metrics_by_method_and_dim[dim]
                results_with_metrics = []
                
                for method_key, ef_values in metrics_by_method.items():
                    avg_ef = statistics.mean(ef_values)
                    std_ef = statistics.stdev(ef_values) if len(ef_values) > 1 else 0.0
                    results_with_metrics.append({
                        'method': method_key,
                        'avg_ef': avg_ef,
                        'std_ef': std_ef,
                        'n_seeds': len(ef_values),
                        'dim': dim
                    })
                
                # Sort by average EF@1% (descending)
                results_with_metrics.sort(key=lambda x: x['avg_ef'], reverse=True)
                
                print(f"{'Method':<60} {'EF@1%':>15} {'N Seeds':>8}")
                print("-" * 80)
                for result in results_with_metrics:
                    if result['std_ef'] > 0:
                        ef_str = f"{result['avg_ef']:.2f} ± {result['std_ef']:.2f}"
                    else:
                        ef_str = f"{result['avg_ef']:.2f}"
                    print(f"{result['method']:<60} {ef_str:>15} {result['n_seeds']:>8}")
            
            print()  # Final newline
            
            # ========================================================================
            # GENERATE VISUALIZATION FIGURES
            # ========================================================================
            # Find best UMAP method from features representation (across all dims)
            # Collect all results across all dimensions for finding best UMAP
            all_results_with_metrics = []
            for dim in all_dims:
                metrics_by_method = metrics_by_method_and_dim[dim]
                for method_key, ef_values in metrics_by_method.items():
                    import statistics
                    avg_ef = statistics.mean(ef_values)
                    std_ef = statistics.stdev(ef_values) if len(ef_values) > 1 else 0.0
                    all_results_with_metrics.append({
                        'method': method_key,
                        'avg_ef': avg_ef,
                        'std_ef': std_ef,
                        'n_seeds': len(ef_values),
                        'dim': dim
                    })
            
            all_results_with_metrics.sort(key=lambda x: x['avg_ef'], reverse=True)
            
            best_umap = None
            for result in all_results_with_metrics:
                method = result['method']
                if 'features' in method and 'UMAP' in method:
                    # Parse method string to extract details
                    parts = method.split('-')
                    best_umap = {
                        'method': method,
                        'representation': 'features',
                        'dr_method': parts[1] if len(parts) > 1 else 'UMAP',
                        'avg_ef': result['avg_ef']
                    }
                    # Extract hyperparameters
                    for part in parts:
                        if part.startswith('nn'):
                            best_umap['n_neighbors'] = part[2:]
                        elif part.startswith('md'):
                            best_umap['min_dist'] = part[2:]
                    break
            
            # Create output directory for this run
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(workspace, f'status_check_outputs_{timestamp}')
            os.makedirs(output_dir, exist_ok=True)
        else:
            print("No ranking metrics found in completed experiments.")
            print()
            output_dir = workspace
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        # No completed experiments
        output_dir = workspace
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
    
    # ============================================================================
    # EXPORT RESULTS AND ORGANIZE OUTPUT
    # ============================================================================
    # Create output directory if not already created
    if 'output_dir' not in locals():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(workspace, f'status_check_outputs_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
    
    # Export to CSV
    output_file = os.path.join(output_dir, f'status_report_{timestamp}.csv')
    try:
        import csv
        with open(output_file, 'w', newline='') as f:
            fieldnames = ['run_dir', 'seed', 'representation', 'dr_method', 'dim', 'n_neighbors', 'min_dist',
                         'completed', 'has_results', 'has_rankings', 'has_metrics', 'error_type', 'log_file']
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)
        print()
        print("=" * 80)
        print(f"OUTPUT FILES")
        print("=" * 80)
        print(f"✅ Detailed report saved to: {output_file}")
        
        # Move debug log to output directory
        import shutil
        new_debug_log_path = os.path.join(output_dir, f'status_check_debug_{timestamp}.log')
        if os.path.exists(debug_log_path):
            shutil.move(debug_log_path, new_debug_log_path)
            print(f"✅ Debug log saved to: {new_debug_log_path}")
        
        print(f"✅ All outputs saved to: {output_dir}")
        print("=" * 80)
    except Exception as e:
        print(f"\n⚠️  Could not save CSV report: {e}")

if __name__ == '__main__':
    main()
