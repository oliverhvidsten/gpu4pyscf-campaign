import json

from conftest import needs_lewis

from gpu4pyscf_campaign import lewis
from gpu4pyscf_campaign.graphs import intended_edges, perceive_edges
from gpu4pyscf_campaign.structures import load_entries, load_raw_entries


@needs_lewis
def test_build_one_preserves_atoms_and_reproduces_graph(structures_path):
    from pymatgen.core.structure import Molecule

    raw = load_raw_entries(structures_path, ["frag_0", "recomb_151"])
    for mid, entry in raw.items():
        new, row = lewis.build_one(entry)
        assert row["error"] is None, row
        assert new is not None and new["molecule_id"] == mid and "molecule_graph" in new
        mol = Molecule.from_dict(new["molecule"])
        stored = Molecule.from_dict(entry["molecule"])
        assert [s.specie.symbol for s in mol] == [s.specie.symbol for s in stored]
        assert mol.charge == entry["charge"] and mol.spin_multiplicity == entry["spin"]
        assert row["lewis_graph_ok"] is True
        assert perceive_edges(mol) == intended_edges(entry)


@needs_lewis
def test_build_all_parallel_and_sidecars(structures_path, tmp_path):
    raw = load_raw_entries(structures_path)
    out, rows = lewis.build_all(raw, workers=2, progress_every=0)
    assert len(out) == len(raw) == len(rows)           # every molecule emitted (stored fallback on failure)
    assert [e["molecule_id"] for e in out] == list(raw)
    assert all(r["emitted_geometry"] in ("lewis", "stored") for r in rows)
    out_json = str(tmp_path / "lewis.json")
    lewis.write_provenance(rows, lewis.provenance_path(out_json), lewis.DEFAULT_VARIANT)
    prov = lewis.read_provenance(lewis.provenance_path(out_json))
    assert prov["n"] == len(rows) and prov["solver_version"]
    summary = lewis.render_summary(rows, len(out), lewis.DEFAULT_VARIANT, out_json)
    assert "Intended bond graph reproduced" in summary and "### by solver rung" in summary
    # the output is itself a valid structures file
    with open(out_json, "w") as fh:
        json.dump(out, fh)
    assert len(load_entries(out_json)) == len(raw)


def test_build_one_without_graph_records_error(structures_path):
    try:
        import lewis_geometry_solver  # noqa: F401
    except ImportError:
        return
    entry = load_raw_entries(structures_path, ["frag_0"])["frag_0"]
    entry = {k: v for k, v in entry.items() if k != "molecule_graph"}
    new, row = lewis.build_one(entry)
    assert new is None and "molecule_graph" in row["error"]
