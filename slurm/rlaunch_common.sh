# Sourced by the run-rlaunch-*.slurm scripts. Activates the env, exports the
# jobflow/FireWorks config and defines `launch_gpu <gpu_index>`.
#
# Requires `source site.env` before sbatch (the scripts are submitted with
# --export=ALL by slurm/submit.sh, so the variables reach the job).
#
# Knobs (pass with `sbatch --export=ALL,VAR=value ...` or in the environment):
#   BATCH      only pull fireworks with spec.batch == BATCH   (default: any batch)
#   QUERY      raw JSON fworker query, overrides BATCH, e.g. '{"spec.natoms": {"$lte": 12}}'.
#              Pass it through the environment (QUERY='...' sbatch ...), not inside
#              --export=ALL,QUERY=..., because sbatch splits that list on commas.
#   NLAUNCHES  max fireworks per GPU process                  (default 0 = until none READY)
#   CATEGORY   firework category                              (default $CAMPAIGN_CATEGORY)
#   MARGIN     seconds before walltime to stop pulling new FWs (default 3600; the longest
#              molecules take several hours, raise it for long-tail batches)
set -euo pipefail

: "${CAMPAIGN_REPO:?source site.env first}"
: "${CAMPAIGN_LAUNCHPAD:?source site.env first}"
: "${JOBFLOW_CONFIG_FILE:?source site.env first}"
LAUNCH_ROOT=${CAMPAIGN_LAUNCH_ROOT:-${SCRATCH:-$HOME}/gpu4pyscf_launches}

module load "${CONDA_MODULE:-conda}"
eval "$(conda shell.bash hook)"
conda activate "${CONDA_ENV:-qcflow}"     # activate.d/cutensor.sh puts libcutensor on LD_LIBRARY_PATH

export JOBFLOW_CONFIG_FILE
[ -n "${FW_CONFIG_FILE:-}" ] && export FW_CONFIG_FILE
[ -n "${QCFLOW_GPU4PYSCF_ROOT:-}" ] && export PYTHONPATH=$QCFLOW_GPU4PYSCF_ROOT${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-${CPUS_PER_GPU:-32}}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

CATEGORY=${CATEGORY:-${CAMPAIGN_CATEGORY:?set CAMPAIGN_CATEGORY}}
SRUN_EXTRA=${SRUN_EXTRA-"--mem=0"}   # whole-node scripts: let each step use node memory; the 1-GPU script sets it empty
NLAUNCHES=${NLAUNCHES:-0}
MARGIN=${MARGIN:-3600}

# walltime of this job in seconds -> rlaunch --timeout (checked between launches).
walltime_s=$(squeue -h -j "$SLURM_JOB_ID" -o %l | head -n 1 | awk -F'[-:]' '{ if (NF==4) print $1*86400+$2*3600+$3*60+$4; else if (NF==3) print $1*3600+$2*60+$3; else if (NF==2) print $1*60+$2; else print $1 }')
TIMEOUT=$(( walltime_s > MARGIN ? walltime_s - MARGIN : walltime_s / 2 ))

BASE=$LAUNCH_ROOT/$SLURM_JOB_ID
mkdir -p "$BASE"

# FireWorker: category (+ optional batch filter). Written per job so BATCH can vary.
FWORKER=$BASE/fworker.yaml
if [ -n "${QUERY:-}" ]; then
  :                                   # raw JSON query from the environment
elif [ -n "${BATCH:-}" ]; then
  QUERY="{\"spec.batch\": $BATCH}"
else
  QUERY="{}"
fi
cat > "$FWORKER" <<YAML
name: ${SLURM_CLUSTER_NAME:-cluster}_gpu4pyscf_${SLURM_JOB_ID}
category: $CATEGORY
query: '$QUERY'
env: {}
YAML

echo "host=$(hostname) job=$SLURM_JOB_ID walltime=${walltime_s}s timeout=${TIMEOUT}s category=$CATEGORY batch=${BATCH:-any} query=$QUERY nlaunches=$NLAUNCHES"
echo "gpus: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | paste -sd ';')"
python - <<'PY'
import gpu4pyscf.lib.cutensor as ct
print("cuTENSOR:", "active" if ct.contract_engine is None else "MISSING -> gpu4pyscf falls back to cupy.einsum (~2x slower); check LD_LIBRARY_PATH")
PY
python -c "import jobflow, fireworks, gpu4pyscf_campaign, crn_utils.pyscf_calc; print('gpu4pyscf_campaign', gpu4pyscf_campaign.__version__)"

launch_gpu() {
  local g=$1
  local dir=$BASE/gpu$g
  mkdir -p "$dir"
  (
    cd "$dir"
    CUDA_VISIBLE_DEVICES=$g srun --exact -n 1 -c "${SLURM_CPUS_PER_TASK:-32}" --gpus=1 $SRUN_EXTRA \
      rlaunch -l "$CAMPAIGN_LAUNCHPAD" -w "$FWORKER" rapidfire --nlaunches "$NLAUNCHES" --timeout "$TIMEOUT" --local_redirect \
      > "$dir/rlaunch.log" 2>&1
  ) &
}

report_gpu() {
  local g=$1
  echo "--- gpu$g: $(cat "$BASE"/gpu$g/launcher_*/FW_job.out 2>/dev/null | grep -c 'Rocket finished') rockets finished / $(ls -d "$BASE"/gpu$g/launcher_* 2>/dev/null | wc -l) launched; log $BASE/gpu$g/rlaunch.log"
}
