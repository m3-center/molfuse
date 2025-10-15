#!/usr/bin/env python3
"""
UMMBAS v3.0 Master Orchestrator

Executes all experimental phases sequentially:
1. Phase 1: Tyro Hyperparameter Sweep (260 runs)
2. Phase 2: MF Cloud Ablation (60 runs)
3. Phase 3: Cross-Protein Generalization (80 runs)
4. Phase 4: Affinity Cutoff Analysis (40 runs)

Total: ~440 runs
"""

import os
import sys
import json
import glob
import subprocess
from datetime import datetime

# Phase configurations
PHASES = {
    1: {
        "name": "Hyperparameter Sweep",
        "config_generator": "generate_phase1_configs.py",
        "config_dir": "hyperparam_configs_v3_phase1",
        "workspace_dir": "experiment_workspace_v3_phase1",
        "expected_runs": 260,
        "description": "Tyro dimensionality × hyperparameter optimization"
    },
    2: {
        "name": "MF Cloud Ablation",
        "config_generator": "generate_phase2_configs.py",
        "config_dir": "hyperparam_configs_v3_phase2_ablation",
        "workspace_dir": "experiment_workspace_v3_phase2",
        "expected_runs": 60,
        "description": "MF cloud size phase transition validation",
        "requires_best_configs": True
    },
    3: {
        "name": "Cross-Protein Generalization",
        "config_generator": "generate_phase3_configs.py",
        "config_dir": "hyperparam_configs_v3_phase3_generalization",
        "workspace_dir": "experiment_workspace_v3_phase3",
        "expected_runs": 80,
        "description": "Generalization to Pyru and Iso proteins",
        "requires_best_configs": True
    },
    4: {
        "name": "Affinity Cutoff Analysis",
        "config_generator": "generate_phase4_configs.py",
        "config_dir": "hyperparam_configs_v3_phase4_cutoff",
        "workspace_dir": "experiment_workspace_v3_phase4",
        "expected_runs": 40,
        "description": "Cutoff threshold sensitivity analysis",
        "requires_best_configs": True
    }
}


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*80)
    print(text)
    print("="*80 + "\n")


def run_command(cmd, description):
    """Run shell command and report status."""
    print(f"→ {description}")
    print(f"  Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  ✓ Success")
        if result.stdout:
            print(f"  Output: {result.stdout[:200]}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed: {e}")
        if e.stderr:
            print(f"  Error: {e.stderr[:500]}")
        return False


def generate_configs(phase_num):
    """Generate configuration files for a phase."""
    phase = PHASES[phase_num]
    
    print_header(f"Phase {phase_num}: Generating Configurations")
    print(f"Phase: {phase['name']}")
    print(f"Description: {phase['description']}")
    print(f"Expected runs: {phase['expected_runs']}")
    print()
    
    # Run config generator
    generator_script = phase['config_generator']
    
    if not os.path.exists(generator_script):
        print(f"✗ Config generator not found: {generator_script}")
        return False
    
    cmd = ["python3", generator_script]
    success = run_command(cmd, f"Generating Phase {phase_num} configs")
    
    if not success:
        return False
    
    # Verify configs were created
    config_dir = phase['config_dir']
    if not os.path.exists(config_dir):
        print(f"✗ Config directory not created: {config_dir}")
        return False
    
    config_files = glob.glob(os.path.join(config_dir, "*.json"))
    print(f"\n✓ Generated {len(config_files)} configuration files")
    print(f"  Location: {config_dir}/")
    
    return True


def extract_best_configs_from_phase1(workspace_dir):
    """Extract best configurations from Phase 1 results."""
    print_header("Extracting Best Configurations from Phase 1")
    
    if not os.path.exists(workspace_dir):
        print(f"✗ Phase 1 workspace not found: {workspace_dir}")
        print("  Phase 1 must complete before proceeding to Phase 2-4")
        return False
    
    cmd = ["python3", "extract_phase1_best_configs.py",
           "--workspace", workspace_dir,
           "--output", "phase1_best_configs.json"]
    
    success = run_command(cmd, "Extracting best configurations")
    
    if not success:
        return False
    
    # Verify best configs file was created
    if not os.path.exists("phase1_best_configs.json"):
        print("✗ Best configs file not created")
        return False
    
    # Display best configs
    with open("phase1_best_configs.json", 'r') as f:
        best_configs = json.load(f)
    
    print(f"\n✓ Extracted {len(best_configs)} best configurations:")
    for key, config in best_configs.items():
        ef = config.get('ef_1_pct_mean', 0)
        print(f"  - {key}: EF@1% = {ef:.2f}")
    
    return True


def run_phase(phase_num):
    """Execute a complete experimental phase."""
    phase = PHASES[phase_num]
    
    print_header(f"Phase {phase_num}: {phase['name']}")
    print(f"Description: {phase['description']}")
    print(f"Expected runs: {phase['expected_runs']}")
    print(f"Workspace: {phase['workspace_dir']}")
    print()
    
    # Create workspace directory
    os.makedirs(phase['workspace_dir'], exist_ok=True)
    
    # Run experiments using existing orchestrator
    config_dir = phase['config_dir']
    config_files = sorted(glob.glob(os.path.join(config_dir, "*.json")))
    
    if not config_files:
        print(f"✗ No configuration files found in {config_dir}")
        return False
    
    print(f"Found {len(config_files)} configurations to run")
    print()
    
    # TODO: Integrate with existing main_orchestrator.py
    # For now, provide instructions
    print("="*80)
    print("MANUAL STEP REQUIRED")
    print("="*80)
    print()
    print(f"Please run the following command to execute Phase {phase_num}:")
    print()
    print(f"  python main_orchestrator.py \\")
    print(f"    --config_dir {config_dir} \\")
    print(f"    --workspace {phase['workspace_dir']} \\")
    print(f"    --n_jobs <NUM_PARALLEL>")
    print()
    print("After Phase 1 completes, run this script again to proceed to Phase 2.")
    print("="*80)
    
    return True


def check_phase_completion(phase_num):
    """Check if a phase has completed."""
    phase = PHASES[phase_num]
    workspace = phase['workspace_dir']
    
    if not os.path.exists(workspace):
        return False, 0
    
    # Count completed runs (those with ranking metrics)
    metrics_pattern = os.path.join(workspace, "run_*/*/results/*/dim_*/*/*_ranking_metrics.csv")
    metrics_files = glob.glob(metrics_pattern)
    
    completed = len(metrics_files)
    expected = phase['expected_runs']
    
    return completed >= expected, completed


def main():
    """Main orchestration logic."""
    print("="*80)
    print("UMMBAS v3.0 - Master Orchestrator")
    print("="*80)
    print()
    print("Experimental Pipeline:")
    for phase_num, phase in PHASES.items():
        print(f"  Phase {phase_num}: {phase['name']} ({phase['expected_runs']} runs)")
    print()
    print(f"Total expected runs: {sum(p['expected_runs'] for p in PHASES.values())}")
    print()
    
    # Determine current phase
    current_phase = None
    for phase_num in sorted(PHASES.keys()):
        is_complete, count = check_phase_completion(phase_num)
        if not is_complete:
            current_phase = phase_num
            break
    
    if current_phase is None:
        print("✓ All phases completed!")
        print_header("Generating Final Report")
        print("Run: python aggregate_and_report.py --workspace_v3")
        return
    
    print(f"Current phase: {current_phase} - {PHASES[current_phase]['name']}")
    print()
    
    # Check if we need to extract best configs from Phase 1
    if current_phase > 1 and not os.path.exists("phase1_best_configs.json"):
        if not extract_best_configs_from_phase1(PHASES[1]['workspace_dir']):
            print("\n✗ Failed to extract best configs from Phase 1")
            print("  Phase 1 must complete successfully before proceeding")
            sys.exit(1)
    
    # Generate configs for current phase
    if not generate_configs(current_phase):
        print(f"\n✗ Failed to generate configs for Phase {current_phase}")
        sys.exit(1)
    
    # Provide instructions to run the phase
    run_phase(current_phase)


if __name__ == "__main__":
    main()
