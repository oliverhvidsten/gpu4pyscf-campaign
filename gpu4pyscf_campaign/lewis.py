"""Lewis-solver starting geometries for a whole structures file.

Each entry is handed to ``lewis_geometry_solver.build`` as is (``molecule_graph``
+ charge + spin; the stored ``molecule`` is the solver's last-resort fallback).
Atom order is preserved, so the output file has the same schema and ids as the
input and can be fed to ``gpu4pyscf-campaign load``. A sidecar
``<out>.provenance.json`` records, per molecule, the solver rung, builder and
whether the stored / Lewis coordinates reproduce the intended bond graph
(OpenBabelNN); ``load`` picks that sidecar up automatically and copies the
solver provenance into the firework metadata.

Validated on the FSI/PF6 set (2026-10-03, 500 molecules): Lewis starts reproduce
the intended graph for 99.8% of molecules (stored coordinates: 17%), halve the
optimizer steps and cut repeat-run basin disagreement from 26% to 5%.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import Any

from gpu4pyscf_campaign.graphs import intended_edges, perceive_edges

logger = logging.getLogger(__name__)

PROV_DROP = {"repaired_graph"}  # large; everything else is kept
DEFAULT_VARIANT = "lewis_graph_listrip"  # = lewis_geometry_solver.LISTRIP

_VARIANT = None  # resolved once per worker process


def provenance_path(structures_out: str) -> str:
    stem = structures_out[:-5] if structures_out.endswith(".json") else structures_out
    return stem + ".provenance.json"


def summary_path(structures_out: str) -> str:
    stem = structures_out[:-5] if structures_out.endswith(".json") else structures_out
    return stem + ".summary.md"


def _clean(prov: dict) -> dict:
    out = {}
    for k, v in prov.items():
        if k in PROV_DROP:
            continue
        try:
            json.dumps(v)
            out[k] = v
        except TypeError:
            out[k] = str(v)
    return out


def _init_worker(variant_name: str) -> None:
    global _VARIANT  # noqa: PLW0603
    import lewis_geometry_solver as lgs  # noqa: PLC0415

    _VARIANT = lgs.VARIANTS[variant_name]


def build_one(entry: dict, variant_name: str | None = None) -> tuple[dict | None, dict]:
    """Build the Lewis start for one entry.

    Returns ``(new_entry_or_None, row)``: the entry with ``molecule`` replaced
    (None when the build failed), and a provenance row.
    """
    import lewis_geometry_solver as lgs  # noqa: PLC0415
    from pymatgen.core.structure import Molecule  # noqa: PLC0415

    variant = _VARIANT if variant_name is None and _VARIANT is not None else lgs.VARIANTS[variant_name or DEFAULT_VARIANT]
    mid = entry["molecule_id"]
    stored = Molecule.from_dict(entry["molecule"])
    intended = intended_edges(entry)
    stored_edges = perceive_edges(stored) if intended is not None else None
    row: dict[str, Any] = {
        "molecule_id": mid,
        "natoms": len(stored),
        "charge": int(entry["charge"]),
        "spin": int(entry["spin"]),
        "stored_graph_ok": None if (stored_edges is None or intended is None) else stored_edges == intended,
    }
    t1 = time.perf_counter()
    new_entry = None
    try:
        if intended is None:
            raise ValueError("entry has no molecule_graph; the Lewis solver needs the intended connectivity")
        mol, prov = lgs.build(copy.deepcopy(entry), variant=variant)
        prov = _clean(prov)
        if [s.specie.symbol for s in mol] != [s.specie.symbol for s in stored]:
            raise RuntimeError("atom order changed")
        mol.set_charge_and_spin(charge=int(entry["charge"]), spin_multiplicity=int(entry["spin"]))
        lewis_edges = perceive_edges(mol)
        row.update(prov)
        row["lewis_graph_ok"] = None if lewis_edges is None else lewis_edges == intended
        row["error"] = None
        new_entry = {k: v for k, v in entry.items() if k != "molecule"}
        new_entry["molecule"] = mol.as_dict()
    except Exception as exc:  # noqa: BLE001 -- record and continue
        row.update({"error": f"{type(exc).__name__}: {exc}", "lewis_graph_ok": None})
    row["build_s"] = time.perf_counter() - t1
    return new_entry, row


def _build_one_star(entry: dict) -> tuple[dict | None, dict]:
    return build_one(entry)


def build_all(
    raw_entries: dict[str, dict],
    variant_name: str = DEFAULT_VARIANT,
    workers: int = 1,
    keep_stored_on_failure: bool = True,
    progress_every: int = 500,
) -> tuple[list[dict], list[dict]]:
    """Build Lewis starts for every entry (in the given order).

    ``keep_stored_on_failure``: when the solver raises, emit the entry with its
    stored coordinates (so the campaign still covers every molecule; the row's
    ``error`` and ``emitted_geometry == "stored"`` record it). Otherwise failed
    molecules are left out of the output.
    """
    entries = list(raw_entries.values())
    out_entries: list[dict] = []
    rows: list[dict] = []
    t0 = time.perf_counter()

    if workers > 1:
        pool = Pool(workers, initializer=_init_worker, initargs=(variant_name,))
        results = pool.imap(_build_one_star, entries, chunksize=20)
    else:
        _init_worker(variant_name)
        pool = None
        results = (build_one(e) for e in entries)

    try:
        for i, (entry, (new_entry, row)) in enumerate(zip(entries, results), 1):
            if new_entry is not None:
                # the solver may itself fall back to the stored coordinates (builder == "stored")
                row["emitted_geometry"] = "stored" if row.get("builder") == "stored" else "lewis"
                out_entries.append(new_entry)
            elif keep_stored_on_failure:
                row["emitted_geometry"] = "stored"
                out_entries.append(entry)
            else:
                row["emitted_geometry"] = None
            rows.append(row)
            if progress_every and (i % progress_every == 0 or i == len(entries)):
                logger.info("  %d/%d built (%.0f s)", i, len(entries), time.perf_counter() - t0)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return out_entries, rows


def write_provenance(rows: list[dict], path: str, variant_name: str) -> None:
    import lewis_geometry_solver as lgs  # noqa: PLC0415

    with open(path, "w") as fh:
        json.dump(
            {"solver_version": lgs.__version__, "variant": variant_name, "n": len(rows), "rows": rows},
            fh, indent=1, default=str,
        )


def read_provenance(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _rate(rows, key):
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    return (sum(1 for v in vals if v) / len(vals)) if vals else None


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(c) for c in r) + " |")
    return "\n".join(lines)


def render_summary(rows: list[dict], n_emitted: int, variant_name: str, out_json: str) -> str:
    import lewis_geometry_solver as lgs  # noqa: PLC0415

    def block(keyfn, title):
        groups = defaultdict(list)
        for r in rows:
            groups[str(keyfn(r))].append(r)
        return (
            f"### by {title}",
            md_table(
                ["group", "n", "stored graph ok", "Lewis graph ok", "errors", "emitted as Lewis"],
                [[g, len(v), _rate(v, "stored_graph_ok"), _rate(v, "lewis_graph_ok"),
                  sum(1 for r in v if r.get("error")), sum(1 for r in v if r.get("emitted_geometry") == "lewis")]
                 for g, v in sorted(groups.items())],
            ),
        )

    rungs = Counter(str(r.get("lewis_rung")) for r in rows)
    builders = Counter(str(r.get("builder")) for r in rows)
    n_err = sum(1 for r in rows if r.get("error"))
    n_stored = sum(1 for r in rows if r.get("emitted_geometry") == "stored")
    lewis_rate = _rate(rows, "lewis_graph_ok")
    stored_rate = _rate(rows, "stored_graph_ok")
    lines = [
        "# Lewis starting geometries", "",
        f"Solver lewis_geometry_solver {lgs.__version__}, variant {variant_name}; {len(rows)} molecules, "
        f"{n_emitted} emitted to {os.path.basename(out_json)} ({n_err} build errors, {n_stored} emitted with stored coordinates). "
        f"Mean build time {sum(r['build_s'] for r in rows) / max(len(rows), 1):.3f} s.", "",
        "**Intended bond graph reproduced (OpenBabelNN):** "
        f"stored coordinates {_fmt(stored_rate)}, Lewis coordinates {_fmt(lewis_rate)}.", "",
        f"Solver rungs: {dict(rungs)}", "",
        f"Builders: {dict(builders)}", "",
    ]
    for keyfn, title in (
        (lambda r: r.get("lewis_rung"), "solver rung"),
        (lambda r: r.get("builder"), "builder"),
        (lambda r: r.get("charge"), "charge"),
        (lambda r: r.get("spin"), "spin multiplicity"),
    ):
        h, t = block(keyfn, title)
        lines += [h, "", t, ""]
    errs = [r for r in rows if r.get("error")]
    if errs:
        lines += ["### Build errors", "", md_table(
            ["molecule_id", "natoms", "charge", "spin", "error"],
            [[r["molecule_id"], r["natoms"], r["charge"], r["spin"], r["error"]] for r in errs[:500]]), ""]
        if len(errs) > 500:
            lines.append(f"... {len(errs) - 500} more in the provenance file")
    return "\n".join(lines)
