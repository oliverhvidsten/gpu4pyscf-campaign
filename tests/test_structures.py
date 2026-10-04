import copy
import json

import pytest

from gpu4pyscf_campaign.structures import (
    load_entries, load_raw_entries, scan_entries, sort_key, validate_entry, write_entries,
)


def test_sort_key_order():
    ids = ["recomb_10", "frag_2", "recomb_9", "frag_10", "other_1", "zzz"]
    assert sorted(ids, key=sort_key) == ["frag_2", "frag_10", "recomb_9", "recomb_10", "other_1", "zzz"]


def test_load_entries_sorted_and_typed(structures_path):
    entries = load_entries(structures_path)
    assert [e.molecule_id for e in entries] == ["frag_0", "frag_12", "frag_13", "recomb_5", "recomb_151"]
    fsi = entries[0]
    assert fsi.natoms == 9 and fsi.charge == -1 and fsi.spin_multiplicity == 1 and fsi.prefix == "frag"
    assert fsi.molecule.charge == -1 and fsi.molecule.spin_multiplicity == 1
    assert entries[3].spin_multiplicity == 2 and entries[3].charge == -2


def test_load_entries_filters(structures_path):
    assert [e.molecule_id for e in load_entries(structures_path, prefix="frag")] == ["frag_0", "frag_12", "frag_13"]
    assert [e.molecule_id for e in load_entries(structures_path, ids=["recomb_5"])] == ["recomb_5"]
    assert len(load_entries(structures_path, limit=2)) == 2
    with pytest.raises(KeyError):
        load_entries(structures_path, ids=["nope_1"])


def test_scan_and_raw(structures_path):
    rows = scan_entries(structures_path)
    assert rows[0]["elements"] == ["F", "N", "O", "S"] and rows[1]["natoms"] == 1
    raw = load_raw_entries(structures_path, ["frag_0"])
    assert set(raw) == {"frag_0"} and "molecule_graph" in raw["frag_0"]


def test_validate_entry_rejects_bad_input(structures_path):
    good = load_raw_entries(structures_path, ["frag_0"])["frag_0"]
    validate_entry(good)
    bad = copy.deepcopy(good)
    bad["spin"] = 2  # FSI- has an even electron count -> doublet impossible
    with pytest.raises(ValueError, match="incompatible"):
        validate_entry(bad)
    bad = copy.deepcopy(good)
    del bad["charge"]
    with pytest.raises(ValueError, match="lacks keys"):
        validate_entry(bad)


def test_roundtrip_and_duplicates(structures_path, tmp_path):
    raw = list(load_raw_entries(structures_path).values())
    out = tmp_path / "copy.json"
    write_entries(raw, str(out))
    assert len(load_entries(str(out))) == len(raw)
    write_entries(raw + [raw[0]], str(out))
    with pytest.raises(ValueError, match="duplicate"):
        load_entries(str(out))
    out.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ValueError, match="JSON list"):
        load_entries(str(out))
