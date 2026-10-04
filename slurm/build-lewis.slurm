#!/bin/bash -l
# CPU job: Lewis starting geometries for a whole structures file (the 87k-molecule FSI set needs
# ~10 GB RAM for json.load and ~0.2 s/molecule/core; 64 cores -> ~5 min of building).
#
#   slurm/submit.sh slurm/build-lewis.slurm --partition=standard /path/structures_fsi.json /path/structures_fsi_lewis.json
#
#SBATCH --job-name=g4p-build-lewis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=02:00:00
set -euo pipefail
: "${CAMPAIGN_REPO:?source site.env first}"
module load "${CONDA_MODULE:-conda}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-qcflow}"
IN=${1:?input structures JSON}; OUT=${2:?output structures JSON}; shift 2
gpu4pyscf-campaign build-lewis "$IN" -o "$OUT" --workers "${SLURM_CPUS_PER_TASK:-8}" "$@"
