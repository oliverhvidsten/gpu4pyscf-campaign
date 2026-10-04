from gpu4pyscf_campaign.fireworks_utils import entry_to_workflow, load_calc_config, make_maker
from gpu4pyscf_campaign.settings import REPO_ROOT
from gpu4pyscf_campaign.structures import load_entries


def test_entry_to_workflow_spec(structures_path):
    calc_config = load_calc_config(str(REPO_ROOT / "config" / "calc_config.yaml"))
    assert calc_config["scf_protocol"] == "soscf_first" and calc_config["max_scf_cycles"] == 50
    maker = make_maker(calc_config, tags=["t"], fmax=0.01, max_steps=500)
    frag, recomb = load_entries(structures_path, ids=["frag_0", "recomb_5"])
    wf = entry_to_workflow(frag, maker, batch=3, category="cat_x", project="proj",
                           extra_metadata={"run_label": "lbl", "starting_geometry": "lewis"})
    assert len(wf.fws) == 1 and wf.name == "frag_0"
    spec = wf.fws[0].spec
    assert spec["_category"] == "cat_x" and spec["_priority"] == 10 and spec["batch"] == 3
    assert spec["molecule_id"] == "frag_0" and spec["natoms"] == 9 and spec["project"] == "proj"
    assert spec["run_label"] == "lbl" and spec["starting_geometry"] == "lewis" and spec["tags"] == ["t"]
    task = wf.fws[0].tasks[0].to_dict()
    assert task["job"]["@module"] == "jobflow.core.job"
    assert task["job"]["function"]["@callable"] == "GPU4PySCFOptMaker.make"
    assert task["job"]["function"]["@module"] == "gpu4pyscf_campaign.jobs"
    assert task["job"]["function_args"][0]["@class"] == "Molecule"
    assert task["job"]["function"]["@bound"]["calc_config"] == calc_config  # the Maker is serialized with the job
    assert task["store"] is None  # resolved from JOBFLOW_CONFIG_FILE at run time: no credentials in the spec
    wf2 = entry_to_workflow(recomb, maker, batch=0)
    assert wf2.fws[0].spec["_priority"] == 0


def test_jobs_module_has_no_gpu_imports():
    import sys

    import gpu4pyscf_campaign.jobs  # noqa: F401

    assert "cupy" not in sys.modules and "gpu4pyscf" not in sys.modules
