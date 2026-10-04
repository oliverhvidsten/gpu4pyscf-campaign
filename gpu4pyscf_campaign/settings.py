"""Site configuration, taken from environment variables so that the same
values serve the Python tools and the Slurm scripts (both read ``site.env``).

Variable                  default                              used by
------------------------  -----------------------------------  -------------------------------
CAMPAIGN_PROJECT          gpu4pyscf_campaign                   metadata.project on specs/docs
CAMPAIGN_CATEGORY         gpu4pyscf_campaign                   FW spec._category / FireWorker
CAMPAIGN_LAUNCHPAD        ~/fw_config/my_launchpad.yaml        load, status, rlaunch
JOBFLOW_CONFIG_FILE       (jobflow default)                    JobStore (task docs) at run time + status
CAMPAIGN_CALC_CONFIG      <repo>/config/calc_config.yaml       calc settings stored in every job
QCFLOW_GPU4PYSCF_ROOT     (unset: rely on the import path)     fallback sys.path for ``crn_utils``
CAMPAIGN_LAUNCH_ROOT      $SCRATCH/gpu4pyscf_launches or cwd   smoke-test output dir
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return os.path.expandvars(os.path.expanduser(value)) if value else default


PROJECT: str = _env("CAMPAIGN_PROJECT", "gpu4pyscf_campaign")
FW_CATEGORY: str = _env("CAMPAIGN_CATEGORY", "gpu4pyscf_campaign")
DEFAULT_LAUNCHPAD: str = _env("CAMPAIGN_LAUNCHPAD", os.path.expanduser("~/fw_config/my_launchpad.yaml"))
DEFAULT_JOBFLOW_CONFIG: str | None = _env("JOBFLOW_CONFIG_FILE")
DEFAULT_CALC_CONFIG: str = _env("CAMPAIGN_CALC_CONFIG", str(REPO_ROOT / "config" / "calc_config.yaml"))
QCFLOW_REPO: str | None = _env("QCFLOW_GPU4PYSCF_ROOT")
LAUNCH_ROOT: str = _env(
    "CAMPAIGN_LAUNCH_ROOT",
    os.path.join(_env("SCRATCH") or _env("PSCRATCH") or os.getcwd(), "gpu4pyscf_launches"),
)
