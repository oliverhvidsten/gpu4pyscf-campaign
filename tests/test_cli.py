import json

from conftest import needs_lewis

from gpu4pyscf_campaign.cli import main


def test_inspect(structures_path, tmp_path, capsys):
    ids_out = tmp_path / "ids.json"
    assert main(["inspect", structures_path, "--ids-out", str(ids_out)]) == 0
    out = capsys.readouterr().out
    assert "5 entries" in out and "single atoms (run as single points): 2" in out
    assert json.load(open(ids_out))[0] == "frag_0"


def test_load_dry_run(structures_path, capsys):
    assert main(["load", structures_path, "--dry-run", "--category", "cat_test", "--run-label", "lbl",
                 "--meta", "site=test", "--batch-size", "2"]) == 0
    out = capsys.readouterr().out
    assert "5 to add" in out and "_category=cat_test" in out and "run_label=lbl" in out
    assert "batches 0..2 (size 2)" in out and "starting_geometry=stored" in out


@needs_lewis
def test_build_lewis_then_load_reads_provenance(structures_path, tmp_path, capsys):
    out_json = str(tmp_path / "lewis.json")
    rc = main(["build-lewis", structures_path, "-o", out_json, "--workers", "1"])
    assert rc in (0, 3)
    assert (tmp_path / "lewis.provenance.json").exists() and (tmp_path / "lewis.summary.md").exists()
    capsys.readouterr()
    assert main(["load", out_json, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "using Lewis provenance for 5 molecules" in out and "starting_geometry=lewis" in out
