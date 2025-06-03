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

python tests/standalone_tsne_test.py \
    --input_csv /home/ahagg2s/UMMBAS_screening_experiments/experiment_workspace/run_20250531_105000/IsocitrateDehydrogenaseNADP_O75874/similarity_spaces/features/dim_2/IsocitrateDehydrogenaseNADP_O75874_features_dim2_similarity_space.csv \
    --descriptor_prefix fp_ \
    --pca_components 50 \
    --perplexity 30 \
    --use_cuml \
    --plot
