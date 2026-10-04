# gpu4pyscf-campaign

Run a whole set of molecules through the GPU geometry optimizer of
[qcflow-gpu4pyscf](https://github.com/rohithsrinivaas/qcflow-gpu4pyscf) (`crn_utils.pyscf_calc`,
wB97M-V/def2-SVPD/SMD by default) on a Slurm cluster, with jobflow + FireWorks for bookkeeping:

```
structures.json  --build-lewis-->  structures_lewis.json  --load-->  FireWorks launchpad (MongoDB)
                                                                          |
                                      task docs + trajectories  <--rlaunch per GPU--  Slurm GPU nodes
                                      (jobflow JobStore: MongoDB + GridFS)
```

* **Input**: a JSON list of `{"molecule_id", "charge", "spin", "molecule", "molecule_graph"}` entries
  (pymatgen `Molecule` / `MoleculeGraph` dicts; `spin` is the multiplicity). This is the
  `structures_fsi.json` format of the FSI/PF6 set (87,292 molecules, 1-30 atoms, charges -2..+2).
* **Starting geometries**: built from the intended bond graph by
  [lewis_geometry_solver](https://github.com/oliverhvidsten/lewis_geometry_solver) (LISTRIP).
  On the FSI/PF6 set this reproduces the intended connectivity for 99.8% of molecules (stored
  coordinates: 17%), halves the optimizer steps and cuts repeat-run basin disagreement from 26% to 5%.
* **One molecule = one jobflow Job = one Firework**; `rlaunch rapidfire` runs them sequentially per
  GPU. Single atoms run a single point instead of an optimization.
* **Output**: `GPU4PySCFTaskDoc` documents (emmet `MoleculeMetadata` + energies, optimizer and SCF
  diagnostics, settings, provenance) in MongoDB; trajectories in GridFS; the ASE `.traj` and
  stdout of every molecule in its launch directory on scratch.

The settings in `config/calc_config.yaml` and the SCF protocol were validated against reference
settings on a 500-molecule sample (median |dE| 0.007 kcal/mol, no bias, 2x faster) and are the
production defaults of the FSI/PF6 campaign.

## Layout

| path | purpose |
| --- | --- |
| `gpu4pyscf_campaign/structures.py` | reads/validates/sorts the structures JSON |
| `gpu4pyscf_campaign/lewis.py` | Lewis starting geometries for a whole file (multiprocess), provenance + summary |
| `gpu4pyscf_campaign/jobs.py` | `GPU4PySCFOptMaker`, the jobflow Maker; imports gpu4pyscf only inside the job |
| `gpu4pyscf_campaign/schemas.py` | `GPU4PySCFTaskDoc` |
| `gpu4pyscf_campaign/fireworks_utils.py` | entry -> Job -> Firework; idempotent loading |
| `gpu4pyscf_campaign/cli.py` | `gpu4pyscf-campaign inspect / build-lewis / load / status / smoke-test` |
| `gpu4pyscf_campaign/settings.py` | site defaults from the environment (`site.env`) |
| `config/calc_config.yaml` | production calculation settings (stored in every firework and task doc) |
| `config/*.example` | launchpad / jobflow / FireWorks config templates (fill in, keep out of git) |
| `site.env.example` | the one site file: env name, paths, launchpad, category, Slurm account/partition |
| `environment.yaml`, `requirements-cuda.txt`, `setup_env.sh` | reproducible environment |
| `slurm/` | `submit.sh` wrapper, whole-node and 1-GPU `rlaunch` scripts, smoke test, CPU Lewis build |
| `tests/` | pytest suite with a 5-molecule fixture (`tests/fixtures/structures_small.json`) |

## 1. Environment (login node)

```bash
git clone <this repo> gpu4pyscf-campaign && cd gpu4pyscf-campaign
cp site.env.example site.env        # edit: allocation, paths, conda module, category
source site.env
bash setup_env.sh                   # conda env + CUDA wheels + cuTENSOR hook + crn_utils + Lewis solver + this package
```

`environment.yaml` is an exact export of the env that ran the campaign on Perlmutter (python 3.12,
gpu4pyscf-cuda12x 1.7.6, cupy-cuda12x 14.1.1, pyscf 2.14, sella 2.5, jobflow 0.3.1, FireWorks 2.1.4,
atomate2 0.1.5, pymatgen 2026.5, rdkit, openbabel). `requirements-cuda.txt` adds the CUDA 12 runtime
wheels and cuTENSOR that the gpu4pyscf wheels need when the machine provides no system CUDA.
Things to check on a new machine:

* `nvidia-smi` on a GPU node: the driver must support CUDA 12.x for the `cu12` wheels. If not,
  switch `gpu4pyscf-cuda12x` / `cupy-cuda12x` / the `nvidia-*-cu12` wheels to the matching stack.
* **cuTENSOR**: cupy links `libcutensor` at load time, so its directory has to be on
  `LD_LIBRARY_PATH` *before* python starts; `setup_env.sh` installs a conda activation hook for
  that. Every Slurm job prints `cuTENSOR: active`; `MISSING` means ~2x slower runs.
* `crn_utils` is a namespace package with no `pyproject`; `setup_env.sh` clones the pinned commit
  into `$QCFLOW_GPU4PYSCF_ROOT` and registers it with a `.pth` file. **The pinned commit
  (`0638ffc`, branch `scf_guess` + SCF continuation) must exist on the remote**; set
  `QCFLOW_URL`/`QCFLOW_REF` if it lives in a fork. Older commits reject the `max_scf_cycles` /
  `scf_protocol` keys of `calc_config.yaml`.
* The launchpad and JobStore MongoDB must be reachable from the compute nodes.

Then copy and fill in `config/my_launchpad.yaml.example`, `config/jobflow.yaml.example` and
`config/FW_config.yaml.example` to the paths named in `site.env` (`chmod 600`).

## 2. Starting geometries

```bash
gpu4pyscf-campaign inspect structures.json                       # counts by prefix/size/charge/spin/element
gpu4pyscf-campaign build-lewis structures.json -o structures_lewis.json --workers 32
#   or as a CPU job:  slurm/submit.sh slurm/build-lewis.slurm --partition=standard structures.json structures_lewis.json
```

`build-lewis` writes the same schema with `molecule` replaced by the Lewis geometry (atom order
preserved), plus `structures_lewis.provenance.json` (per molecule: solver rung, builder, whether the
stored / Lewis coordinates reproduce the intended graph, build errors) and
`structures_lewis.summary.md`. Molecules whose build fails are emitted with their stored
coordinates (`--drop-failed` leaves them out instead); exit code 3 flags that errors occurred.
The whole file is `json.load`ed (the 737 MB FSI file needs ~10 GB RAM and ~30 s), then built in
parallel at ~0.2 s per molecule per core.

## 3. Load fireworks

```bash
gpu4pyscf-campaign load structures_lewis.json --dry-run --limit 3        # inspect specs, no DB writes
gpu4pyscf-campaign load structures_lewis.json --run-label lewis_prod --tag fsi_pf6_2026-10
```

* One Firework per molecule, `_category` = `$CAMPAIGN_CATEGORY`, `_priority` 10 for `frag_*` else 0,
  `spec.batch = ordinal // 200` over the sorted list (`frag_*` then `recomb_*`, numerically).
* Idempotent: ids already in the launchpad under the same category are skipped; batch numbers are
  stable across reruns.
* The spec/task-doc metadata carries `molecule_id, charge, spin_multiplicity, natoms, batch, project,
  run_label, tags` and, when a `build-lewis` provenance sidecar sits next to the structures file,
  `starting_geometry` (`lewis`/`stored`), `lewis_rung`, `lewis_builder`, `lewis_graph_ok`,
  `lewis_solver_version`. `--meta KEY=VALUE` adds anything else. No credentials: the JobStore is
  resolved on the compute node from `JOBFLOW_CONFIG_FILE`.
* Use a **fresh category** per machine/campaign: a FireWorker pulls every READY firework of its
  category in the shared launchpad.

## 4. Run

```bash
source site.env
slurm/submit.sh slurm/smoke-test.slurm                              # 1 GPU, fixture molecules, no FireWorks
slurm/submit.sh slurm/run-rlaunch-gpu.slurm                         # 1 node = 4 rlaunch processes, one per H100
slurm/submit.sh slurm/run-rlaunch-gpu.slurm --array=0-9             # 10 nodes draining the same pool
BATCH=7 slurm/submit.sh slurm/run-rlaunch-gpu.slurm                 # only spec.batch == 7
NLAUNCHES=5 slurm/submit.sh slurm/run-rlaunch-gpu-1.slurm -t 01:00:00
```

`submit.sh` adds `--account`, `--partition`, `--output` from `site.env` and exports the environment;
options before the script name go to `sbatch`, arguments after it to the script. Knobs (environment):
`BATCH`, `QUERY` (raw FireWorker JSON query), `NLAUNCHES` (per GPU process, 0 = until none READY),
`CATEGORY`, `MARGIN` (seconds before walltime to stop pulling fireworks, default 3600; the longest
molecules take hours, so raise it for long-tail batches). Each job writes
`$CAMPAIGN_LAUNCH_ROOT/<jobid>/{fworker.yaml, gpu<N>/rlaunch.log, gpu<N>/launcher_*/}`; the
per-molecule `FW_job.out` (Sella steps, SCF fallbacks) and `.traj` live in the launcher dirs.

Sizing: on A100-40GB with cuTENSOR, 10-18-atom molecules take 2-10 min each (median 27 optimizer
steps from Lewis starts); a few hundred of 87k take > 2 h. The Slurm scripts assume Kestrel's H100
nodes (4 GPUs, 128 cores -> 4 tasks x 32 cores, `--gres=gpu:h100:N`, 2-day limit); adjust the
`#SBATCH` lines for another cluster.

## 5. Monitor

```bash
gpu4pyscf-campaign status                                # FW states per batch for $CAMPAIGN_CATEGORY
gpu4pyscf-campaign status --fizzled                      # exceptions + rerun command
gpu4pyscf-campaign status --doc recomb_5 [--full]        # task doc summary (JOBFLOW_CONFIG_FILE)
gpu4pyscf-campaign status --unconverged --run-label lewis_prod
lpad -l $CAMPAIGN_LAUNCHPAD rerun_fws -i <fw_id ...>
```

Python access to results:

```python
import os; os.environ["JOBFLOW_CONFIG_FILE"] = "/path/to/jobflow_campaign.yaml"
from jobflow import SETTINGS
store = SETTINGS.JOB_STORE; store.connect()
from monty.json import MontyDecoder
doc = store.query_one({"metadata.molecule_id": "frag_0", "metadata.run_label": "lewis_prod"}, load=True)
out = MontyDecoder().process_decoded(doc["output"])     # -> Molecule / Trajectory objects
out["energy_ha"], out["molecule"], out["trajectory"], out["state"], out["n_opt_steps"]
```

Task-doc fields of note: `state` (`successful` / `unconverged` = hit `max_steps`), `energy_ha` (on
`final_grid`), `n_opt_steps`, `n_scf`, `n_scf_restarts`, `n_soscf_fallbacks`, `scf_restart_log`,
`spin_contamination`, `settings` (as used), `calc_config` (as passed), `gpu_name`,
`elapsed_time_s`, `crn_utils_git_sha`, `dir_name`.

## Caveats

* Spin multiplicities are passed through unchanged from the input file.
* `pyscf_calc`'s own code defaults are the old slow settings: `calc_config.yaml` must stay complete.
* No frequency / thermochemistry step; energies are electronic (SMD) only.
* GPU runs are not bit-reproducible (atomicAdd reductions; Sella amplifies the noise), so repeat
  runs can land in different basins (~5% with Lewis starts). Not fixable by settings.
* Scratch is purged; everything essential is in MongoDB / GridFS.

## Tests

```bash
python -m pytest -q          # 14 tests; the Lewis ones skip without lewis_geometry_solver/rdkit/openbabel
```
