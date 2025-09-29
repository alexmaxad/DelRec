#!/bin/bash
#SBATCH --job-name=1_day      # Name of the job
#SBATCH --output=outs_slurm/outputs/output.txt.%j     # Output file (with job ID)
#SBATCH --error=outs_slurm/errors/error.txt.%j       # Error file (with job ID)
#SBATCH --ntasks=1                 # Number of tasks (processes)
#SBATCH --cpus-per-task=16         # Number of CPUs per task
#SBATCH --mem=64G                  # Allocated memory
#SBATCH --time=72:00:00            # Time limit (HH:MM:SS)
#SBATCH --partition=gpu            # Partition to use
#SBATCH --gres=gpu:1

conda init
conda activate py310

export WANDB_MODE=offline
export TQDM_DISABLE=1

python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA version:', torch.version.cuda)"
python main_perf_SSC.py

echo "Job executed on $(hostname)"