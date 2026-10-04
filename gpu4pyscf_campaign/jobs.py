"""jobflow Maker wrapping ``crn_utils.pyscf_calc`` (qcflow-gpu4pyscf).

The heavy GPU imports happen inside :meth:`GPU4PySCFOptMaker.make` so that
this module can be imported on login nodes (to build/serialize fireworks) and
so that ``CUDA_VISIBLE_DEVICES`` set by the launcher is honoured.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Maker, job

from gpu4pyscf_campaign import __version__
from gpu4pyscf_campaign.schemas import GPU4PySCFTaskDoc
from gpu4pyscf_campaign.settings import QCFLOW_REPO

if TYPE_CHECKING:
    from pymatgen.core.structure import Molecule

logger = logging.getLogger(__name__)

_DATA_OBJECTS = ["trajectory"]
REQUIRED_CALC_KEYS = ("xc", "basis")


def _import_pyscf_calc():
    """Import ``crn_utils.pyscf_calc`` (qcflow-gpu4pyscf).

    ``crn_utils`` is a namespace package without a pyproject, so it is made
    importable either by a ``.pth`` file / ``PYTHONPATH`` (setup_env.sh does the
    former) or, as a fallback, by ``QCFLOW_GPU4PYSCF_ROOT``.
    """
    try:
        from crn_utils import pyscf_calc  # noqa: PLC0415
    except ModuleNotFoundError:
        if not QCFLOW_REPO:
            raise ModuleNotFoundError(
                "crn_utils (qcflow-gpu4pyscf) is not importable; install it (setup_env.sh) or set "
                "QCFLOW_GPU4PYSCF_ROOT to the checkout"
            ) from None
        if QCFLOW_REPO not in sys.path:
            sys.path.insert(0, QCFLOW_REPO)
        from crn_utils import pyscf_calc  # noqa: PLC0415
    return pyscf_calc


def _crn_utils_repo(pyscf_calc) -> str:
    """The qcflow-gpu4pyscf checkout that ``pyscf_calc`` was imported from."""
    return os.path.dirname(os.path.dirname(os.path.abspath(pyscf_calc.__file__)))


def _git_sha(repo: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=20,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 -- provenance only
        return None


def _gpu_name() -> str | None:
    try:
        import cupy  # noqa: PLC0415

        props = cupy.cuda.runtime.getDeviceProperties(cupy.cuda.runtime.getDevice())
        name = props["name"]
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:  # noqa: BLE001
        return None


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for key, mod in (("gpu4pyscf_version", "gpu4pyscf"), ("pyscf_version", "pyscf")):
        try:
            out[key] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            out[key] = None
    return out


def _ase_traj_to_pmg(traj_path: str, charge: int, spin_multiplicity: int):
    """ASE .traj (one frame per optimizer step, with energy/forces) ->
    pymatgen Trajectory with per-frame ``energy_ev`` / ``forces_ev_per_ang``."""
    from ase.io import read as ase_read  # noqa: PLC0415
    from pymatgen.core.structure import Molecule  # noqa: PLC0415
    from pymatgen.core.trajectory import Trajectory  # noqa: PLC0415

    frames = ase_read(traj_path, index=":")
    # Sella (internal=True) appends dummy atoms (symbol "X") when an angle goes linear, so
    # later frames can have more atoms than earlier ones. Keep the real atoms only (they
    # come first and their count never changes); otherwise Trajectory.from_molecules fails
    # and the document is stored without a trajectory.
    n_real = min(sum(1 for s in a.get_chemical_symbols() if s != "X") for a in frames)
    mols, props = [], []
    for atoms in frames:
        # Atoms read from a .traj carry no charge/spin, and AseAtomsAdaptor.get_molecule
        # would then assume a neutral singlet (invalid for odd-electron systems).
        mol = Molecule(
            atoms.get_chemical_symbols()[:n_real], atoms.get_positions()[:n_real],
            charge=charge, spin_multiplicity=spin_multiplicity,
        )
        mols.append(mol)
        frame: dict = {}
        try:
            frame["energy_ev"] = float(atoms.get_potential_energy())
            frame["forces_ev_per_ang"] = atoms.get_forces()[:n_real].tolist()
        except Exception:  # noqa: BLE001 -- frame written without a calculator
            pass
        if len(atoms) > n_real:
            frame["n_dummy_atoms"] = len(atoms) - n_real
        props.append(frame)
    return Trajectory.from_molecules(mols, frame_properties=props)


@dataclass
class GPU4PySCFOptMaker(Maker):
    """Geometry optimization (Sella + analytic GPU Hessian) with GPU4PySCF.

    Parameters
    ----------
    calc_config
        Full settings block for ``pyscf_calc`` (``xc``, ``basis``, ``smd_solvent``
        and the speed-up keys ``grid``, ``final_grid``, ``nlc_grid``,
        ``scf_conv_tol``, ``scf_guess``, ``reuse_mf``). REQUIRED: the function's
        own defaults are the old, slow, seeded-guess settings.
    fmax
        Sella max-force convergence criterion in eV/Angstrom.
    max_steps
        Optimizer step limit; exceeding it yields ``state="unconverged"``.
    store_trajectory
        Convert the ASE trajectory to a pymatgen Trajectory and store it in the
        ``data`` store (the ``.traj`` file stays in the launch directory too).
    tags
        Free-form tags copied into the task document.
    """

    name: str = "gpu4pyscf geometry optimization"
    calc_config: dict = field(default_factory=dict)
    fmax: float = 0.01
    max_steps: int = 500
    store_trajectory: bool = True
    tags: list[str] = field(default_factory=list)

    @job(data=_DATA_OBJECTS, output_schema=GPU4PySCFTaskDoc)
    def make(
        self,
        molecule: Molecule,
        molecule_id: str,
        charge: int | None = None,
        spin_multiplicity: int | None = None,
    ) -> GPU4PySCFTaskDoc:
        """Optimize ``molecule`` (or run a single point if it is one atom)."""
        missing = [k for k in REQUIRED_CALC_KEYS if k not in self.calc_config]
        if missing:
            raise ValueError(
                f"calc_config is missing {missing}; pass the full input_params block "
                "(pyscf_calc defaults are the old slow settings)"
            )
        charge = molecule.charge if charge is None else int(charge)
        spin_multiplicity = (
            molecule.spin_multiplicity if spin_multiplicity is None else int(spin_multiplicity)
        )
        molecule = molecule.copy()
        molecule.set_charge_and_spin(charge=charge, spin_multiplicity=spin_multiplicity)

        pyscf_calc = _import_pyscf_calc()
        calc_config = dict(self.calc_config)
        cwd = os.getcwd()
        t0 = time.perf_counter()

        if molecule.num_sites == 1:
            task_type = "single point"
            result = pyscf_calc.single_point(molecule, charge, spin_multiplicity, calc_config)
            traj_path = None
        else:
            task_type = "geometry optimization"
            traj_path = os.path.join(cwd, f"{molecule_id}.traj")
            result = pyscf_calc.optimize_and_analyze(
                molecule, charge, spin_multiplicity, calc_config,
                fmax=self.fmax, max_steps=self.max_steps, trajectory_path=traj_path,
            )
        elapsed = time.perf_counter() - t0

        trajectory = None
        n_frames = None
        if self.store_trajectory and traj_path and os.path.exists(traj_path):
            try:
                trajectory = _ase_traj_to_pmg(traj_path, charge, spin_multiplicity)
                n_frames = len(trajectory)
            except Exception:  # noqa: BLE001 -- the result matters more than the trajectory
                logger.warning("Could not convert %s to a pymatgen Trajectory", traj_path, exc_info=True)

        from atomate2.utils.datetime import datetime_str  # noqa: PLC0415

        # emmet's point-group analysis of a single atom spams numpy ComplexWarnings
        import warnings  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415

        warnings.filterwarnings("ignore", category=np.exceptions.ComplexWarning)
        doc = GPU4PySCFTaskDoc.from_result(
            result,
            molecule_id=molecule_id,
            task_type=task_type,
            calc_config=calc_config,
            fmax=self.fmax if task_type == "geometry optimization" else None,
            max_steps=self.max_steps if task_type == "geometry optimization" else None,
            trajectory=trajectory,
            n_trajectory_frames=n_frames,
            elapsed_time_s=elapsed,
            dir_name=cwd,
            hostname=socket.gethostname(),
            gpu_name=_gpu_name(),
            completed_at=datetime_str(),
            tags=list(self.tags),
            campaign_version=__version__,
            crn_utils_git_sha=_git_sha(_crn_utils_repo(pyscf_calc)),
            **_versions(),
        )
        logger.info(
            "%s %s natoms=%d charge=%d mult=%d steps=%d converged=%s restarts=%d E=%.6f Ha wall=%.0fs",
            task_type, molecule_id, molecule.num_sites, charge, spin_multiplicity,
            doc.n_opt_steps, doc.converged, doc.n_scf_restarts, doc.energy_ha, elapsed,
        )
        return doc
