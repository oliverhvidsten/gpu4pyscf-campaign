#!/bin/bash -l
# Single-GPU variant (a quarter of a Kestrel H100 node; charged per GPU) for small batches
# and tests. Same knobs as run-rlaunch-gpu.slurm (BATCH, QUERY, NLAUNCHES, CATEGORY, MARGIN).
#
#   NLAUNCHES=5 slurm/submit.sh slurm/run-rlaunch-gpu-1.slurm -t 02:00:00
#
#SBATCH --job-name=g4p-rlaunch-1gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=90G
#SBATCH --time=06:00:00

SRUN_EXTRA=""   # a shared allocation cannot ask for --mem=0 (all node memory) in its step
source "${CAMPAIGN_REPO:?source site.env first}/slurm/rlaunch_common.sh"
launch_gpu 0
wait
report_gpu 0
