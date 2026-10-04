"""Turn structure entries into FireWorks Workflows (one Firework per molecule)
and keep loading idempotent against the launchpad."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import yaml
from jobflow.managers.fireworks import flow_to_workflow

from gpu4pyscf_campaign.jobs import GPU4PySCFOptMaker
from gpu4pyscf_campaign.settings import FW_CATEGORY, PROJECT

if TYPE_CHECKING:
    from fireworks import LaunchPad, Workflow

    from gpu4pyscf_campaign.structures import Entry

# Firework priority by molecule_id prefix (higher runs first); anything else gets 0.
PREFIX_PRIORITY = {"frag": 10}


def load_calc_config(path: str) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict) or "xc" not in cfg or "basis" not in cfg:
        raise ValueError(f"{path} must be a flat mapping with at least xc and basis")
    return cfg


def make_maker(calc_config: dict, tags: list[str] | None = None, **kwargs) -> GPU4PySCFOptMaker:
    return GPU4PySCFOptMaker(calc_config=calc_config, tags=list(tags or []), **kwargs)


def entry_to_workflow(
    entry: Entry,
    maker: GPU4PySCFOptMaker,
    batch: int,
    category: str = FW_CATEGORY,
    project: str = PROJECT,
    extra_metadata: dict | None = None,
    priority: int | None = None,
) -> Workflow:
    """One jobflow Job -> one Firework -> one Workflow.

    ``store=None`` so the JobStore is resolved at run time from
    ``JOBFLOW_CONFIG_FILE`` (no credentials in the firework spec).

    The metadata dict is copied by jobflow into both the firework spec and the
    task document's ``metadata``; ``extra_metadata`` (run_label, Lewis-solver
    provenance, ...) is how runs are told apart later.
    """
    job = maker.make(
        entry.molecule,
        molecule_id=entry.molecule_id,
        charge=entry.charge,
        spin_multiplicity=entry.spin_multiplicity,
    )
    job.name = f"gpu4pyscf_opt {entry.molecule_id}"
    metadata = {
        "molecule_id": entry.molecule_id,
        "charge": entry.charge,
        "spin_multiplicity": entry.spin_multiplicity,
        "natoms": entry.natoms,
        "id_prefix": entry.prefix,
        "batch": int(batch),
        "project": project,
        "category": category,
        "tags": list(maker.tags),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    job.update_metadata(metadata)
    job.update_config(
        {
            "manager_config": {
                "_category": category,
                "_priority": PREFIX_PRIORITY.get(entry.prefix, 0) if priority is None else int(priority),
            }
        }
    )
    return flow_to_workflow(
        job,
        store=None,
        name=entry.molecule_id,
        metadata={"molecule_id": entry.molecule_id, "batch": int(batch), "project": project},
    )


def ensure_indexes(lpad: LaunchPad) -> None:
    lpad.fireworks.create_index("spec.molecule_id", background=True)
    lpad.fireworks.create_index([("spec._category", 1), ("state", 1)], background=True)
    lpad.fireworks.create_index([("spec._category", 1), ("spec.batch", 1)], background=True)


def existing_molecule_ids(lpad: LaunchPad, category: str = FW_CATEGORY) -> set[str]:
    return set(lpad.fireworks.distinct("spec.molecule_id", {"spec._category": category}))


def chunked(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def fizzled_rows(lpad: LaunchPad, category: str, ids: Iterable[str] | None = None) -> list[dict]:
    """FIZZLED fireworks of a category with the last traceback line of their last launch."""
    query: dict = {"spec._category": category, "state": "FIZZLED"}
    if ids is not None:
        query["spec.molecule_id"] = {"$in": list(ids)}
    rows = []
    for fw in lpad.fireworks.find(query, {"fw_id": 1, "spec.molecule_id": 1, "launches": 1}):
        msg = ""
        if fw.get("launches"):
            launch = lpad.launches.find_one({"launch_id": fw["launches"][-1]}, {"action.stored_data._exception": 1})
            exc = (((launch or {}).get("action") or {}).get("stored_data") or {}).get("_exception", {})
            tb = exc.get("_stacktrace", "") if isinstance(exc, dict) else str(exc)
            lines = [ln for ln in str(tb).strip().splitlines() if ln.strip()]
            msg = lines[-1] if lines else ""
        rows.append({"fw_id": fw["fw_id"], "molecule_id": fw.get("spec", {}).get("molecule_id"), "error": msg})
    return rows
