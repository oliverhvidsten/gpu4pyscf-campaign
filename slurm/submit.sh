#!/bin/bash
# sbatch wrapper that injects the site settings (account, partition, log dir) and exports
# the environment to the job, so the .slurm files carry no site-specific directives.
#
#   source site.env
#   slurm/submit.sh slurm/run-rlaunch-gpu.slurm                              # 4 GPUs, drain the category
#   BATCH=3 slurm/submit.sh slurm/run-rlaunch-gpu.slurm                      # only spec.batch == 3
#   slurm/submit.sh slurm/run-rlaunch-gpu.slurm --array=0-9                  # 10 nodes on the same pool
#   NLAUNCHES=5 slurm/submit.sh slurm/run-rlaunch-gpu-1.slurm -t 01:00:00    # 1 GPU, short test
#   slurm/submit.sh slurm/smoke-test.slurm --ids frag_0,frag_12              # args after the script go to the script
set -euo pipefail
: "${CAMPAIGN_REPO:?source site.env first}"
: "${SBATCH_ACCOUNT:?set SBATCH_ACCOUNT in site.env}"
script=$1; shift
mkdir -p "${CAMPAIGN_LOG_DIR:-$CAMPAIGN_REPO/logs}"
sbatch_args=()
while [ $# -gt 0 ] && [[ $1 == -* ]]; do           # leading -x / --xx[=v] options go to sbatch
  sbatch_args+=("$1")
  if [[ $1 != *=* && $1 == -? ]]; then shift; sbatch_args+=("$1"); fi   # short option with a separate value
  shift
done
partition_args=(--partition="${GPU_PARTITION:-gpu-h100}")
for a in "${sbatch_args[@]}"; do [[ $a == --partition* || $a == -p ]] && partition_args=(); done   # caller chose one
exec sbatch --account="$SBATCH_ACCOUNT" "${partition_args[@]}" \
  --output="${CAMPAIGN_LOG_DIR:-$CAMPAIGN_REPO/logs}/%x-%j.out" --error="${CAMPAIGN_LOG_DIR:-$CAMPAIGN_REPO/logs}/%x-%j.out" \
  --export=ALL "${sbatch_args[@]}" "$script" "$@"
