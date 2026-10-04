#!/bin/bash -l
# GPU smoke test of the jobflow job without FireWorks: `gpu4pyscf-campaign smoke-test` runs a few
# molecules with run_locally and an in-memory JobStore, writes <molecule_id>.json per result.
#
#   slurm/submit.sh slurm/smoke-test.slurm                                   # fixture molecules (FSI-, F-, a 14-atom dianion)
#   slurm/submit.sh slurm/smoke-test.slurm /path/structures.json --ids frag_0,frag_12
#
#SBATCH --job-name=g4p-smoke
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=90G
#SBATCH --time=00:45:00
set -euo pipefail
: "${CAMPAIGN_REPO:?source site.env first}"
module load "${CONDA_MODULE:-conda}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-qcflow}"
[ -n "${QCFLOW_GPU4PYSCF_ROOT:-}" ] && export PYTHONPATH=$QCFLOW_GPU4PYSCF_ROOT${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
python -c 'import gpu4pyscf.lib.cutensor as ct; print("cuTENSOR:", "active" if ct.contract_engine is None else "MISSING")'
STRUCTURES=${1:-$CAMPAIGN_REPO/tests/fixtures/structures_small.json}; [ $# -gt 0 ] && shift
srun --ntasks=1 --gpus=1 gpu4pyscf-campaign smoke-test "$STRUCTURES" \
  --outdir "${CAMPAIGN_LAUNCH_ROOT:-${SCRATCH:-$HOME}/gpu4pyscf_launches}/smoke/$SLURM_JOB_ID" "$@"
