"""Read molecule sets in the ``structures_fsi.json`` format.

The file is a JSON list; each entry is::

    {"molecule_id": "frag_0",            # unique string id, ideally "<prefix>_<int>"
     "charge": -1,
     "spin": 1,                          # spin multiplicity (2S+1)
     "molecule": {...},                  # pymatgen Molecule as_dict()
     "molecule_graph": {...}}            # pymatgen MoleculeGraph as_dict(): the INTENDED bonding

``molecule`` is what gets optimized (unless the Lewis builder replaces it);
``molecule_graph`` is what the Lewis builder works from and is dropped from the
firework specs to keep them small. ``spin`` wins over ``molecule.spin_multiplicity``
when they disagree.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from pymatgen.core.structure import Molecule

REQUIRED_KEYS = ("molecule_id", "charge", "spin", "molecule")
# ids of the form "<prefix>_<int>" sort by this prefix order, then numerically;
# anything else sorts after them alphabetically.
PREFIX_ORDER = {"frag": 0, "recomb": 1}
_ID_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*)_(\d+)$")


@dataclass
class Entry:
    molecule_id: str
    charge: int
    spin_multiplicity: int
    molecule: Molecule

    @property
    def natoms(self) -> int:
        return self.molecule.num_sites

    @property
    def prefix(self) -> str:
        return self.molecule_id.split("_", 1)[0]


def sort_key(molecule_id: str) -> tuple[int, str, int, str]:
    m = _ID_RE.match(molecule_id)
    if not m:
        return (9, "", 0, molecule_id)
    prefix, idx = m.group(1), int(m.group(2))
    return (PREFIX_ORDER.get(prefix, 5), prefix, idx, "")


def validate_entry(d: dict, path: str = "<json>") -> None:
    """Raise ValueError with a readable message if ``d`` is not a valid entry."""
    missing = [k for k in REQUIRED_KEYS if k not in d]
    if missing:
        raise ValueError(f"{path}: entry {d.get('molecule_id', '?')!r} lacks keys {missing}")
    mol = d["molecule"]
    if not isinstance(mol, dict) or "sites" not in mol:
        raise ValueError(f"{path}: entry {d['molecule_id']!r}: 'molecule' must be a pymatgen Molecule dict")
    if not mol["sites"]:
        raise ValueError(f"{path}: entry {d['molecule_id']!r} has no atoms")
    try:
        charge, spin = int(d["charge"]), int(d["spin"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: entry {d['molecule_id']!r}: charge/spin must be integers") from exc
    if spin < 1:
        raise ValueError(f"{path}: entry {d['molecule_id']!r}: spin multiplicity {spin} < 1")
    n_el = sum(_ATOMIC_NUMBERS.get(s["species"][0]["element"], 0) for s in mol["sites"]) - charge
    if n_el and (n_el - (spin - 1)) % 2:
        raise ValueError(
            f"{path}: entry {d['molecule_id']!r}: {n_el} electrons is incompatible with multiplicity {spin}"
        )


def _read(path: str) -> list[dict]:
    """json.load the whole file (the 87k-molecule FSI set is 737 MB, ~30 s, several GB of RAM)."""
    with open(path) as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a JSON list of entries, got {type(raw).__name__}")
    seen: set[str] = set()
    for d in raw:
        validate_entry(d, path)
        if d["molecule_id"] in seen:
            raise ValueError(f"{path}: duplicate molecule_id {d['molecule_id']!r}")
        seen.add(d["molecule_id"])
    return raw


def load_entries(
    path: str,
    ids: Iterable[str] | None = None,
    prefix: str | None = None,
    limit: int | None = None,
) -> list[Entry]:
    """Load (a filtered subset of) the structures as :class:`Entry`, sorted with :func:`sort_key`."""
    wanted = set(ids) if ids is not None else None
    entries: list[Entry] = []
    for d in _read(path):
        mid = d["molecule_id"]
        if wanted is not None and mid not in wanted:
            continue
        if prefix is not None and not mid.startswith(prefix + "_"):
            continue
        mol = Molecule.from_dict(d["molecule"])
        entries.append(Entry(molecule_id=mid, charge=int(d["charge"]), spin_multiplicity=int(d["spin"]), molecule=mol))
    entries.sort(key=lambda e: sort_key(e.molecule_id))
    if limit is not None:
        entries = entries[:limit]
    if wanted is not None:
        missing = wanted - {e.molecule_id for e in entries}
        if missing:
            raise KeyError(f"molecule_ids not found in {path}: {sorted(missing)[:10]}")
    return entries


def iter_ids(path: str) -> Iterator[str]:
    for d in _read(path):
        yield d["molecule_id"]


def _elements(entry: dict) -> list[str]:
    return sorted({site["species"][0]["element"] for site in entry["molecule"]["sites"]})


def scan_entries(path: str) -> list[dict]:
    """Lightweight rows for every entry (no Molecule objects), sorted with
    :func:`sort_key`: ``molecule_id, prefix, natoms, charge, spin, elements``."""
    rows = []
    for d in _read(path):
        mid = d["molecule_id"]
        rows.append(
            {
                "molecule_id": mid,
                "prefix": mid.split("_", 1)[0],
                "natoms": len(d["molecule"]["sites"]),
                "charge": int(d["charge"]),
                "spin": int(d["spin"]),
                "elements": _elements(d),
            }
        )
    rows.sort(key=lambda r: sort_key(r["molecule_id"]))
    return rows


def load_raw_entries(path: str, ids: Iterable[str] | None = None) -> dict[str, dict]:
    """The untouched entry dicts (``molecule`` AND ``molecule_graph``) keyed by
    molecule_id, for the requested ids (all entries if ``ids`` is None)."""
    wanted = set(ids) if ids is not None else None
    out = {d["molecule_id"]: d for d in _read(path) if wanted is None or d["molecule_id"] in wanted}
    if wanted is not None:
        missing = wanted - set(out)
        if missing:
            raise KeyError(f"molecule_ids not found in {path}: {sorted(missing)[:10]}")
    return out


def write_entries(entries: list[dict], path: str) -> None:
    with open(path, "w") as fh:
        json.dump(entries, fh)


_ATOMIC_NUMBERS = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56,
}
