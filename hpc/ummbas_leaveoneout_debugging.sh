#!/bin/bash
#SBATCH --partition=gpu          # partition / wait queue
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks-per-node=32     # number of tasks per node
#SBATCH --mem=128G               # memory per node in MB (different units with suffix K|M|G|T)
#SBATCH --time=24:00:00              # total runtime of job allocation (format D-HH:MM:SS; first parts optional)
#SBATCH --output=slurm.%j.out    # filename for STDOUT (%N: nodename, %j: job-ID)
#SBATCH --error=slurm.%j.err     # filename for STDERR
#SBATCH --gres=gpu:1

module load nvidia-hpc/default
module load cuda
source /home/ahagg2s/miniforge3/bin/activate ummbas-screening
python main_orchestrator.py --config debugging_config.json 
# python main_orchestrator.py --config experiment_config.json 
