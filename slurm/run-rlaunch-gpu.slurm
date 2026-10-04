#!/bin/bash -l
# Drain fireworks on one whole GPU node: one `rlaunch rapidfire` per GPU, each running
# molecules sequentially in a single Python process (CUDA context reused).
# Kestrel H100 nodes: 4 GPUs, 128 cores -> 4 tasks x 32 cores. Max walltime 2 days.
#
#   source site.env
#   slurm/submit.sh slurm/run-rlaunch-gpu.slurm                 # all READY fireworks in the category
#   BATCH=3 slurm/submit.sh slurm/run-rlaunch-gpu.slurm         # only spec.batch == 3
#   slurm/submit.sh slurm/run-rlaunch-gpu.slurm --array=0-9     # 10 nodes draining the same pool
#   slurm/submit.sh slurm/run-rlaunch-gpu.slurm -t 2-00:00:00 MARGIN=10800   (long-tail batches)
#
#SBATCH --job-name=g4p-rlaunch
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gres=gpu:h100:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00

source "${CAMPAIGN_REPO:?source site.env first}/slurm/rlaunch_common.sh"
NGPU=${GPUS_PER_NODE:-4}
for ((g = 0; g < NGPU; g++)); do launch_gpu "$g"; done
wait
echo "all rlaunch processes exited"
for ((g = 0; g < NGPU; g++)); do report_gpu "$g"; done
