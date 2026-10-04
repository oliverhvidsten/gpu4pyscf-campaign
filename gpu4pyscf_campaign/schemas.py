"""Task document for a GPU4PySCF geometry optimization (or single point).

Follows the atomate2 convention: an emmet ``MoleculeMetadata`` subclass whose
composition/formula fields are filled from the optimized molecule, plus the
fields reported by ``crn_utils.pyscf_calc.optimize_and_analyze`` /
``single_point``.
"""

from __future__ import annotations

from typing import Any, Literal

from emmet.core.structure import MoleculeMetadata
from pydantic import Field
from pymatgen.core.structure import Molecule
from pymatgen.core.trajectory import Trajectory

HARTREE_TO_EV = 27.211386245988

TaskType = Literal["geometry optimization", "single point"]
TaskState = Literal["successful", "unconverged"]


class GPU4PySCFTaskDoc(MoleculeMetadata):
    """Output of one ``GPU4PySCFOptMaker`` job."""

    molecule_id: str = Field(..., description="Identifier from the input structures JSON")
    task_type: TaskType = Field(..., description="geometry optimization, or single point for monoatomic species")
    state: TaskState = Field(..., description="'unconverged' if the optimizer hit max_steps without reaching fmax")

    initial_molecule: Molecule = Field(..., description="Input geometry")
    molecule: Molecule = Field(..., description="Final (optimized) geometry")

    energy_ha: float = Field(..., description="Final electronic energy in Hartree (on final_grid if configured)")
    energy_ev: float = Field(..., description="energy_ha in eV")
    converged: bool = Field(..., description="Optimizer convergence flag (True for single points)")
    n_opt_steps: int = Field(0, description="Optimizer steps taken")
    n_scf_restarts: int = Field(0, description="SCFs that only converged after a fallback")
    scf_restart_log: list[str] = Field(default_factory=list, description="Which fallback rescued each restarted SCF")
    n_guess_rejections: int = Field(0, description="Seeded SCFs redone cold by the energy guard")
    n_scf: int | None = Field(None, description="SCFs requested during the optimization (pyscf_calc >= 2026-10-02)")
    n_soscf_fallbacks: int | None = Field(None, description="SCFs that needed the second-order (SOSCF/Newton) fallback")
    opt_wall_time_s: float | None = Field(None, description="Wall time of the optimizer loop alone (s)")
    final_state_check: dict[str, Any] | None = Field(None, description="Cold-SCF wrong-state check at the final geometry (None with scf_guess=atomic)")
    spin_contamination: dict[str, Any] | None = Field(None, description="<S^2> analysis for open-shell species; None for closed-shell")
    point_group: str | None = None
    rotational_symmetry_number: int | float | None = None

    xc: str | None = None
    basis: str | None = None
    smd_solvent: str | None = None
    calc_config: dict[str, Any] = Field(default_factory=dict, description="calc_config dict passed to pyscf_calc (frozen input)")
    settings: dict[str, Any] = Field(default_factory=dict, description="Settings actually used, as reported by pyscf_calc")
    fmax: float | None = Field(None, description="Force convergence criterion (eV/A)")
    max_steps: int | None = None

    trajectory: Trajectory | None = Field(None, description="Optimization trajectory (energy/forces per frame); stored in the 'data' store")
    n_trajectory_frames: int | None = None

    elapsed_time_s: float | None = None
    dir_name: str | None = None
    hostname: str | None = None
    gpu_name: str | None = None
    completed_at: str | None = None
    tags: list[str] = Field(default_factory=list)

    campaign_version: str | None = None
    gpu4pyscf_version: str | None = None
    pyscf_version: str | None = None
    crn_utils_git_sha: str | None = None

    @classmethod
    def from_result(
        cls,
        result: dict[str, Any],
        *,
        molecule_id: str,
        task_type: TaskType,
        calc_config: dict[str, Any],
        **extra: Any,
    ) -> GPU4PySCFTaskDoc:
        """Build the document from the dict returned by ``optimize_and_analyze``
        or ``single_point`` (keys documented in ``crn_utils/pyscf_calc.py``)."""
        initial = result["initial_molecule"]
        final = result["molecule"]
        initial_mol = initial if isinstance(initial, Molecule) else Molecule.from_dict(initial)
        final_mol = final if isinstance(final, Molecule) else Molecule.from_dict(final)
        energy_ha = float(result["energy_Ha"])
        converged = bool(result.get("converged", True))

        return cls.from_molecule(
            meta_molecule=final_mol,
            molecule_id=molecule_id,
            task_type=task_type,
            state="successful" if converged else "unconverged",
            initial_molecule=initial_mol,
            molecule=final_mol,
            energy_ha=energy_ha,
            energy_ev=energy_ha * HARTREE_TO_EV,
            converged=converged,
            n_opt_steps=int(result.get("n_opt_steps", 0)),
            n_scf_restarts=int(result.get("n_scf_restarts", 0)),
            scf_restart_log=list(result.get("scf_restart_log", [])),
            n_guess_rejections=int(result.get("n_guess_rejections", 0)),
            n_scf=result.get("n_scf"),
            n_soscf_fallbacks=result.get("n_soscf_fallbacks"),
            opt_wall_time_s=result.get("opt_wall_time_s"),
            final_state_check=result.get("final_state_check"),
            spin_contamination=result.get("spin_contamination"),
            point_group=result.get("point_group"),
            rotational_symmetry_number=result.get("rotational_symmetry_number"),
            xc=result.get("xc", calc_config.get("xc")),
            basis=result.get("basis", calc_config.get("basis")),
            smd_solvent=result.get("smd_solvent", calc_config.get("smd_solvent")),
            calc_config=dict(calc_config),
            settings=dict(result.get("settings", {})),
            **extra,
        )
