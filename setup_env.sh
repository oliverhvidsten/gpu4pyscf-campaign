#!/bin/bash
# Build the `qcflow` conda env for this repo on a login node (compute nodes have no network).
#
#   source site.env            # CONDA_MODULE, CONDA_ENV, QCFLOW_GPU4PYSCF_ROOT
#   bash setup_env.sh          # idempotent; re-run after fixing a wheel
#
# Steps: conda env from environment.yaml -> CUDA 12 wheels + cuTENSOR (requirements-cuda.txt)
# -> cuTENSOR activation hook -> qcflow-gpu4pyscf checkout (crn_utils, via a .pth file)
# -> lewis_geometry_solver -> this package (editable). Pinned commits are at the top.
set -euo pipefail

QCFLOW_URL=${QCFLOW_URL:-https://github.com/rohithsrinivaas/qcflow-gpu4pyscf.git}
QCFLOW_REF=${QCFLOW_REF:-0638ffcd5ea0795a0abf8f82ac04da771571970f}   # branch scf_guess + SCF continuation (2026-10-03)
LEWIS_URL=${LEWIS_URL:-https://github.com/oliverhvidsten/lewis_geometry_solver.git}
LEWIS_REF=${LEWIS_REF:-b750e52}                                        # S/P support + no-Lewis-structure rejection

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CONDA_MODULE=${CONDA_MODULE:-conda}
CONDA_ENV=${CONDA_ENV:-qcflow}
QCFLOW_GPU4PYSCF_ROOT=${QCFLOW_GPU4PYSCF_ROOT:-$HERE/../qcflow-gpu4pyscf}

module load "$CONDA_MODULE" 2>/dev/null || true
eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo ">>> creating conda env $CONDA_ENV from environment.yaml (several GB; ~15 min)"
  conda env create -n "$CONDA_ENV" -f "$HERE/environment.yaml"
else
  echo ">>> conda env $CONDA_ENV exists; updating from environment.yaml"
  conda env update -n "$CONDA_ENV" -f "$HERE/environment.yaml" --prune
fi
conda activate "$CONDA_ENV"

echo ">>> CUDA runtime wheels + cuTENSOR"
pip install -r "$HERE/requirements-cuda.txt"

echo ">>> cuTENSOR activation hook (cupy links libcutensor at load time: it must be on LD_LIBRARY_PATH)"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/cutensor.sh" <<'HOOK'
_lib=$(python -c "import cutensor, os; print(os.path.join(os.path.dirname(cutensor.__file__), 'lib'))" 2>/dev/null || true)
if [ -n "$_lib" ] && [ -d "$_lib" ]; then
    export QCFLOW_LD_LIBRARY_PATH_BACKUP="${LD_LIBRARY_PATH-}"
    export LD_LIBRARY_PATH="$_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
unset _lib
HOOK
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/cutensor.sh" <<'HOOK'
if [ -n "${QCFLOW_LD_LIBRARY_PATH_BACKUP+x}" ]; then
    export LD_LIBRARY_PATH="$QCFLOW_LD_LIBRARY_PATH_BACKUP"; unset QCFLOW_LD_LIBRARY_PATH_BACKUP
fi
HOOK

echo ">>> qcflow-gpu4pyscf (crn_utils) at $QCFLOW_REF -> $QCFLOW_GPU4PYSCF_ROOT"
if [ ! -d "$QCFLOW_GPU4PYSCF_ROOT/.git" ]; then
  git clone "$QCFLOW_URL" "$QCFLOW_GPU4PYSCF_ROOT"
fi
git -C "$QCFLOW_GPU4PYSCF_ROOT" fetch --all --quiet
if ! git -C "$QCFLOW_GPU4PYSCF_ROOT" checkout --quiet "$QCFLOW_REF"; then
  echo "!!! $QCFLOW_REF is not in $QCFLOW_URL (unpushed?). Push it, or set QCFLOW_URL/QCFLOW_REF to where it lives." >&2
  exit 1
fi
SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "$QCFLOW_GPU4PYSCF_ROOT" > "$SITE_PACKAGES/qcflow_gpu4pyscf.pth"

echo ">>> lewis_geometry_solver at $LEWIS_REF + this package"
pip install "git+$LEWIS_URL@$LEWIS_REF"
pip install -e "$HERE"

echo ">>> checks"
conda deactivate && conda activate "$CONDA_ENV"
python - <<'PY'
import importlib
for m in ("jobflow", "fireworks", "atomate2", "pymatgen", "rdkit", "openbabel", "lewis_geometry_solver", "gpu4pyscf_campaign"):
    mod = importlib.import_module(m); print(f"  {m:24s} {getattr(mod, '__version__', '')}")
import crn_utils, os; print("  crn_utils from", os.path.dirname(crn_utils.__path__[0]))
PY
echo "GPU-side check (run on a GPU node):  python -c 'import gpu4pyscf.lib.cutensor as c; print(\"cuTENSOR\", \"active\" if c.contract_engine is None else \"MISSING\")'"
echo "done. Next: cp site.env.example site.env; edit; source site.env; sbatch slurm/smoke-test.slurm"
