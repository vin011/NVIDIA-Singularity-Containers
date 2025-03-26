#!/usr/bin/env bash
### Please adjust the following variables accordingly
#SBATCH --job-name=tensorflow-model-gpu
#SBATCH --account=XXXXX
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --array=1-10
#SBATCH --time=04:00:00
#SBATCH --output=pytorch-gpu.o%j.%N

export SINGULARITY_MODULE=singularitypro/3.11
export CONTAINER_DIR=$HOME/NVIDIA-Singularity-Containers/sif_files
export ENV_OUPUT_FILE_NAME=$SLURM_JOB_ID.environment.txt

module reset
module load $SINGUALRITY_MODULE

cd /scratch/$USER/job_$SLURM_JOBID

## Print relevant environment and resource variables
module list >> $ENV_OUTPUT_FILE_NAME
printenv >> $ENV_OUTPUT_FILE_NAME
nvidia-smi >> $ENV_OUTPUT_FILE_NAME

# singularity exec --bind /expanse,/scratch $CONTAINER_DIR/pytorch_24.11-py3.sif 'import torch; print(f"TORCH & CUDA: {torch.cuda.is_available()}")'

declare -xr LUSTRE_PROJECT_DIR="/expanse/lustre/projects/${SLURM_JOB_ACCOUNT}/${USER}"
declare -xr LUSTRE_SCRATCH_DIR="/expanse/lustre/scratch/${USER}/temp_project"
declare -xr LOCAL_SCRATCH_DIR="/scratch/${USER}/job_${SLURM_JOB_ID}"

export TF_USE_LEGACY_KERAS=1
printenv

cd "${SLURM_SUBMIT_DIR}"

echo "Running the training script from ${SLURM_SUBMIT_DIR} ..."
time -p python3 -u tensorflow-model-training.py --classes 10 --precision fp32 --epochs 1 --batch_size 256 --accelerator gpu --savekeras True

echo "Job completed"
