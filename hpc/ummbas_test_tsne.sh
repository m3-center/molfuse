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

python standalone_tsne_test_combined.py \
    --run_type features \
    --chembl_csv /home/ahagg2s/UMMBAS_screening_experiments/experiment_workspace/run_20250531_105000/IsocitrateDehydrogenaseNADP_O75874/temp_data/IsocitrateDehydrogenaseNADP_O75874_chembl_mf_excluded_features.csv \
    --zinc_csv /home/ahagg2s/UMMBAS_screening_experiments/experiment_workspace/run_20250531_105000/IsocitrateDehydrogenaseNADP_O75874/temp_data/IsocitrateDehydrogenaseNADP_O75874_zinc_excluded_features.csv \
    --descriptor_cols "DipoleMoment,ABC,nAcid,nBase,nAromAtom,nAtom,nH,nC,nN,nO,nS,nP,nX,nBonds,nBondsO,nBondsS,nBondsD,nBondsT,nBondsA,nBondsM,nBondsKS,nBondsKD,EState_VSA7,nHBAcc,nHBDon,Lipinski,apol,bpol,nRing,n3Ring,n4Ring,n5Ring,n6Ring,n7Ring,n8Ring,nRot,Diameter,TopoShapeIndex,Vabc,MW" \ 
    --pca_components 50 \
    --perplexity 30 \
    --use_cuml \
    --plot

python standalone_tsne_test_combined.py \
    --run_type fingerprints \
    --chembl_csv /home/ahagg2s/UMMBAS_screening_experiments/experiment_workspace/run_20250531_105000/IsocitrateDehydrogenaseNADP_O75874/temp_data/IsocitrateDehydrogenaseNADP_O75874_chembl_mf_excluded_fingerprints.csv \
    --zinc_csv /home/ahagg2s/UMMBAS_screening_experiments/experiment_workspace/run_20250531_105000/IsocitrateDehydrogenaseNADP_O75874/temp_data/IsocitrateDehydrogenaseNADP_O75874_zinc_excluded_fingerprints.csv \
    --descriptor_prefix "fp_" \
    --pca_components 50 \
    --perplexity 30 \
    --use_cuml \
    --plot