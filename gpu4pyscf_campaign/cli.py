"""Command-line entry point: ``gpu4pyscf-campaign <subcommand>``.

    inspect      summarize a structures JSON (counts by prefix / size / charge / spin / elements)
    build-lewis  structures JSON -> structures JSON with Lewis-solver starting geometries (+ provenance)
    load         structures JSON -> one Firework per molecule in the launchpad (idempotent)
    status       firework states per batch, task docs, unconverged / fizzled lists
    smoke-test   run a few molecules with jobflow run_locally on a GPU node (no FireWorks)

Site defaults come from the environment (``source site.env``; see settings.py).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter

from gpu4pyscf_campaign import settings

logger = logging.getLogger("gpu4pyscf_campaign")


# --------------------------------------------------------------------------- helpers
def _read_ids(args) -> list[str] | None:
    ids = args.ids.split(",") if getattr(args, "ids", None) else None
    if getattr(args, "ids_file", None):
        with open(args.ids_file) as fh:
            loaded = json.load(fh)
        ids = list(loaded[args.ids_key]) if isinstance(loaded, dict) else list(loaded)
        if not ids:
            raise SystemExit(f"{args.ids_file} holds no molecule_ids")
    return ids


def _add_selection(p: argparse.ArgumentParser) -> None:
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--ids", help="comma-separated molecule_ids")
    sel.add_argument("--prefix", help="only ids with this prefix (e.g. frag)")
    sel.add_argument("--ids-file", help="JSON file: a list of molecule_ids, or a dict holding one under --ids-key")
    p.add_argument("--ids-key", default="ids", help="key of the id list inside --ids-file when it is a dict (default %(default)s)")
    p.add_argument("--limit", type=int, help="stop after N molecules (after sorting/filtering)")


def get_store(path: str | None):
    """``path`` is a jobflow settings file (top-level JOB_STORE key), the same
    file JOBFLOW_CONFIG_FILE points at on the compute nodes."""
    import yaml
    from jobflow import JobStore

    if not path:
        raise SystemExit("no jobflow config: set JOBFLOW_CONFIG_FILE or pass --jobflow-config")
    with open(path) as fh:
        spec = yaml.safe_load(fh)["JOB_STORE"]
    store = JobStore.from_dict_spec(spec)
    store.connect()
    return store


# --------------------------------------------------------------------------- inspect
def cmd_inspect(args) -> int:
    from gpu4pyscf_campaign.structures import scan_entries

    t0 = time.perf_counter()
    rows = scan_entries(args.structures)
    print(f"{args.structures}: {len(rows)} entries ({time.perf_counter() - t0:.0f} s)")

    def show(title, counter, limit=20):
        items = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))[:limit]
        print(f"  {title}: " + ", ".join(f"{k}={v}" for k, v in items))

    show("by prefix", Counter(r["prefix"] for r in rows))
    show("by natoms", Counter(r["natoms"] for r in rows), limit=40)
    show("by charge", Counter(r["charge"] for r in rows))
    show("by spin multiplicity", Counter(r["spin"] for r in rows))
    show("elements", Counter(e for r in rows for e in r["elements"]))
    n_single = sum(1 for r in rows if r["natoms"] == 1)
    print(f"  single atoms (run as single points): {n_single}")
    if args.ids_out:
        with open(args.ids_out, "w") as fh:
            json.dump([r["molecule_id"] for r in rows], fh)
        print(f"  wrote sorted id list to {args.ids_out}")
    return 0


# --------------------------------------------------------------------------- build-lewis
def cmd_build_lewis(args) -> int:
    from gpu4pyscf_campaign import lewis
    from gpu4pyscf_campaign.structures import load_raw_entries, sort_key, write_entries

    try:
        import lewis_geometry_solver as lgs  # noqa: F401
    except ModuleNotFoundError:
        raise SystemExit("lewis_geometry_solver is not installed: pip install '.[lewis]' (plus openbabel from conda-forge)") from None

    ids = _read_ids(args)
    t0 = time.perf_counter()
    raw = load_raw_entries(args.structures, ids)
    order = sorted(raw, key=sort_key)
    if args.prefix:
        order = [m for m in order if m.startswith(args.prefix + "_")]
    if args.limit:
        order = order[: args.limit]
    raw = {m: raw[m] for m in order}
    logger.info("loaded %d entries from %s in %.0f s; solver %s variant %s; %d workers",
                len(raw), args.structures, time.perf_counter() - t0, lgs.__version__, args.variant, args.workers)

    out_entries, rows = lewis.build_all(
        raw, variant_name=args.variant, workers=args.workers,
        keep_stored_on_failure=not args.drop_failed,
    )
    if args.drop_graph:
        for e in out_entries:
            e.pop("molecule_graph", None)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_entries(out_entries, args.out)
    lewis.write_provenance(rows, lewis.provenance_path(args.out), args.variant)
    summary = lewis.render_summary(rows, len(out_entries), args.variant, args.out)
    with open(lewis.summary_path(args.out), "w") as fh:
        fh.write(summary)
    print("\n".join(summary.splitlines()[2:8]))
    print(f"wrote {args.out} ({len(out_entries)} entries), {lewis.provenance_path(args.out)}, {lewis.summary_path(args.out)}")
    n_err = sum(1 for r in rows if r.get("error"))
    return 0 if n_err == 0 else 3


# --------------------------------------------------------------------------- load
def cmd_load(args) -> int:
    from gpu4pyscf_campaign.fireworks_utils import (
        chunked, ensure_indexes, entry_to_workflow, existing_molecule_ids, load_calc_config, make_maker,
    )
    from gpu4pyscf_campaign.lewis import provenance_path, read_provenance
    from gpu4pyscf_campaign.structures import load_entries

    calc_config = load_calc_config(args.calc_config)
    extra_metadata: dict = {}
    for item in args.meta:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--meta expects KEY=VALUE, got {item!r}")
        extra_metadata[key] = value
    tags = list(args.tag)
    if args.run_label:
        extra_metadata["run_label"] = args.run_label
        if args.run_label not in tags:
            tags.append(args.run_label)

    # Lewis-solver provenance sidecar written by build-lewis -> per-molecule metadata
    per_mol: dict[str, dict] = {}
    prov = None if args.no_provenance else read_provenance(provenance_path(args.structures))
    if prov:
        extra_metadata.setdefault("lewis_solver_version", prov.get("solver_version"))
        extra_metadata.setdefault("lewis_variant", prov.get("variant"))
        for r in prov["rows"]:
            per_mol[r["molecule_id"]] = {
                "starting_geometry": r.get("emitted_geometry") or "stored",
                "lewis_rung": r.get("lewis_rung"),
                "lewis_builder": r.get("builder"),
                "lewis_graph_ok": r.get("lewis_graph_ok"),
            }
        print(f"using Lewis provenance for {len(per_mol)} molecules from {provenance_path(args.structures)}")
    else:
        extra_metadata.setdefault("starting_geometry", "stored")

    maker = make_maker(calc_config, tags=tags, fmax=args.fmax, max_steps=args.max_steps,
                       store_trajectory=not args.no_trajectory)

    t0 = time.perf_counter()
    entries = load_entries(args.structures, ids=_read_ids(args), prefix=args.prefix, limit=args.limit)
    print(f"loaded {len(entries)} entries from {args.structures} in {time.perf_counter() - t0:.0f} s", flush=True)

    lpad = None
    already: set[str] = set()
    if not args.dry_run:
        from fireworks import LaunchPad

        lpad = LaunchPad.from_file(args.launchpad)
        ensure_indexes(lpad)
        already = existing_molecule_ids(lpad, args.category)
        print(f"launchpad already has {len(already)} fireworks in category {args.category!r}", flush=True)

    todo = [e for e in entries if e.molecule_id not in already]
    print(f"{len(todo)} to add, {len(entries) - len(todo)} skipped as already present", flush=True)
    if not todo:
        return 0

    # `batch` is assigned from the ordinal over the FULL sorted list so batch
    # numbers stay stable across (idempotent) reruns of this loader.
    ordinal = {e.molecule_id: i for i, e in enumerate(entries)}
    n_added = 0
    for chunk in chunked(todo, args.chunk):
        wfs = []
        for e in chunk:
            meta = dict(extra_metadata)
            meta.update(per_mol.get(e.molecule_id, {}))
            wfs.append(entry_to_workflow(
                e, maker, batch=ordinal[e.molecule_id] // args.batch_size, category=args.category,
                project=args.project, extra_metadata=meta or None,
            ))
        if args.dry_run:
            for wf in wfs[:3]:
                fw = wf.fws[0]
                print(f"  [dry-run] wf={wf.name!r} fw={fw.name!r} spec keys={sorted(k for k in fw.spec if not k.startswith('_tasks'))}")
                print(f"            _category={fw.spec.get('_category')} _priority={fw.spec.get('_priority')} batch={fw.spec.get('batch')} "
                      f"natoms={fw.spec.get('natoms')} run_label={fw.spec.get('run_label')} tags={fw.spec.get('tags')} "
                      f"starting_geometry={fw.spec.get('starting_geometry')} lewis_rung={fw.spec.get('lewis_rung')}")
        else:
            lpad.bulk_add_wfs(wfs)
        n_added += len(wfs)
        print(f"  {n_added}/{len(todo)} {'built' if args.dry_run else 'added'} ({time.perf_counter() - t0:.0f} s)", flush=True)

    n_batches = (len(entries) - 1) // args.batch_size + 1
    print(f"done: {n_added} workflows {'built' if args.dry_run else 'added'}; batches 0..{n_batches - 1} (size {args.batch_size})")
    return 0


# --------------------------------------------------------------------------- status
_STATES = ["READY", "RESERVED", "RUNNING", "COMPLETED", "FIZZLED", "PAUSED", "DEFUSED", "WAITING", "ARCHIVED"]


def _counts(lpad, category: str) -> None:
    from collections import defaultdict

    by_batch: dict[int, Counter] = defaultdict(Counter)
    total: Counter = Counter()
    for d in lpad.fireworks.find({"spec._category": category}, {"state": 1, "spec.batch": 1}):
        b = d.get("spec", {}).get("batch", -1)
        by_batch[b][d["state"]] += 1
        total[d["state"]] += 1
    print(f"category {category!r}: {sum(total.values())} fireworks")
    print("  " + " ".join(f"{s}={total.get(s, 0)}" for s in _STATES if total.get(s, 0)))
    if len(by_batch) > 1:
        print(f"{'batch':>6} " + " ".join(f"{s[:5]:>6}" for s in _STATES))
        for b in sorted(by_batch):
            c = by_batch[b]
            if any(c.get(s) for s in _STATES if s != "READY") or len(by_batch) <= 50:
                print(f"{b:>6} " + " ".join(f"{c.get(s, 0):>6}" for s in _STATES))


def _summarize(doc: dict) -> str:
    o = doc.get("output", {}) or {}
    m = doc.get("metadata", {}) or {}
    return (
        f"{o.get('molecule_id')} [{m.get('run_label') or '-'} / {m.get('starting_geometry') or '?'} start]: {o.get('task_type')} "
        f"state={o.get('state')} E={o.get('energy_ha')} Ha steps={o.get('n_opt_steps')} restarts={o.get('n_scf_restarts')} "
        f"natoms={o.get('natoms')} q={o.get('charge')} mult={o.get('spin_multiplicity')} "
        f"wall={o.get('elapsed_time_s') and round(o['elapsed_time_s'])}s gpu={o.get('gpu_name')} "
        f"completed={o.get('completed_at')} fw_id={m.get('fw_id')} dir={o.get('dir_name')}"
    )


def _label_filter(label: str | None) -> dict:
    return {} if label is None else {"metadata.run_label": label}


def cmd_status(args) -> int:
    from gpu4pyscf_campaign.fireworks_utils import fizzled_rows

    if not any([args.counts, args.doc, args.unconverged, args.fizzled]):
        args.counts = True
    if args.counts or args.fizzled:
        from fireworks import LaunchPad

        lpad = LaunchPad.from_file(args.launchpad)
        if args.counts:
            _counts(lpad, args.category)
        if args.fizzled:
            ids = []
            for row in fizzled_rows(lpad, args.category):
                ids.append(row["fw_id"])
                print(f"fw_id={row['fw_id']} {row['molecule_id']}: {row['error'][:160]}")
            print(f"{len(ids)} FIZZLED")
            if ids:
                print(f"rerun with:  lpad -l {args.launchpad} rerun_fws -i " + " ".join(map(str, ids[:50])) + (" ..." if len(ids) > 50 else ""))
    if args.doc or args.unconverged:
        store = get_store(args.jobflow_config)
        if args.doc:
            docs = list(store.query({"metadata.molecule_id": args.doc, **_label_filter(args.run_label)}, load=False))
            if not docs:
                print(f"no task document for {args.doc!r} in {store.docs_store.collection_name}")
            for d in docs:
                print(_summarize(d))
                if args.full:
                    from monty.json import jsanitize

                    d.pop("output", {}).pop("trajectory", None)
                    print(json.dumps(jsanitize(d, strict=False), indent=1, default=str)[:20000])
        if args.unconverged:
            n = 0
            for d in store.query({"output.state": "unconverged", **_label_filter(args.run_label)},
                                 properties=["output", "metadata"], load=False):
                print(_summarize(d))
                n += 1
            print(f"{n} unconverged")
    return 0


# --------------------------------------------------------------------------- smoke-test
def cmd_smoke_test(args) -> int:
    from jobflow import Flow, JobStore, run_locally
    from maggma.stores import MemoryStore
    from monty.json import jsanitize

    from gpu4pyscf_campaign.fireworks_utils import load_calc_config, make_maker
    from gpu4pyscf_campaign.structures import load_entries

    outdir = args.outdir or os.path.join(settings.LAUNCH_ROOT, "smoke", os.environ.get("SLURM_JOB_ID", str(int(time.time()))))
    os.makedirs(outdir, exist_ok=True)
    os.chdir(outdir)
    entries = load_entries(args.structures, ids=_read_ids(args), prefix=args.prefix, limit=args.limit)
    if not entries:
        raise SystemExit("no molecules selected")
    maker = make_maker(load_calc_config(args.calc_config), tags=["smoke_test"])
    jobs = [maker.make(e.molecule, molecule_id=e.molecule_id, charge=e.charge, spin_multiplicity=e.spin_multiplicity) for e in entries]
    for j, e in zip(jobs, entries):
        j.name = f"gpu4pyscf_opt {e.molecule_id}"
        j.update_metadata({"molecule_id": e.molecule_id})

    store = JobStore(MemoryStore(), additional_stores={"data": MemoryStore()})
    t0 = time.perf_counter()
    responses = run_locally(Flow(jobs, name="smoke"), store=store, create_folders=True, ensure_success=True)
    print(f"run_locally finished in {time.perf_counter() - t0:.0f} s; outputs in {outdir}")

    ok = True
    for job, e in zip(jobs, entries):
        doc = responses[job.uuid][1].output
        d = doc.model_dump() if hasattr(doc, "model_dump") else doc
        with open(os.path.join(outdir, f"{e.molecule_id}.json"), "w") as fh:
            json.dump(jsanitize(d, strict=True), fh)
        print(
            f"{e.molecule_id}: {d['task_type']} state={d['state']} E={d['energy_ha']:.6f} Ha steps={d['n_opt_steps']} "
            f"restarts={d['n_scf_restarts']} traj_frames={d.get('n_trajectory_frames')} wall={d['elapsed_time_s']:.0f}s "
            f"gpu={d['gpu_name']} formula={d.get('formula_pretty')} q={d['charge']} mult={d['spin_multiplicity']}"
        )
        ok &= d["state"] == "successful"
    stored = list(store.query({"metadata.molecule_id": entries[0].molecule_id}, load=True))
    if stored:
        from monty.json import MontyDecoder

        traj = MontyDecoder().process_decoded(stored[0]["output"].get("trajectory"))
        print(f"JobStore doc for {entries[0].molecule_id}: trajectory from data store -> {type(traj).__name__} "
              f"({len(traj) if traj is not None else 'n/a'} frames)")
    return 0 if ok else 1


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gpu4pyscf-campaign", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("inspect", help="summarize a structures JSON")
    s.add_argument("structures")
    s.add_argument("--ids-out", help="write the sorted molecule_id list to this JSON file")
    s.set_defaults(func=cmd_inspect)

    s = sub.add_parser("build-lewis", help="Lewis-solver starting geometries for every molecule")
    s.add_argument("structures")
    s.add_argument("-o", "--out", required=True, help="output structures JSON (sidecars: .provenance.json, .summary.md)")
    s.add_argument("--variant", default="lewis_graph_listrip", help="lewis_geometry_solver variant (default %(default)s)")
    s.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2), help="processes (default %(default)s)")
    s.add_argument("--drop-failed", action="store_true",
                   help="leave molecules whose build failed OUT of the output instead of emitting their stored coordinates")
    s.add_argument("--drop-graph", action="store_true", help="omit molecule_graph from the output (smaller file; no graph-fidelity analysis later)")
    _add_selection(s)
    s.set_defaults(func=cmd_build_lewis)

    s = sub.add_parser("load", help="load molecules into the FireWorks launchpad")
    s.add_argument("structures")
    s.add_argument("--calc-config", default=settings.DEFAULT_CALC_CONFIG)
    s.add_argument("--launchpad", default=settings.DEFAULT_LAUNCHPAD)
    s.add_argument("--category", default=settings.FW_CATEGORY, help="FW spec._category (default %(default)s)")
    s.add_argument("--project", default=settings.PROJECT, help="metadata.project (default %(default)s)")
    s.add_argument("--batch-size", type=int, default=200, help="molecules per `batch` spec value (default %(default)s)")
    s.add_argument("--chunk", type=int, default=1000, help="workflows per bulk_add_wfs call (default %(default)s)")
    _add_selection(s)
    s.add_argument("--tag", action="append", default=[], help="tag copied into task docs (repeatable)")
    s.add_argument("--run-label", help="stored as metadata.run_label in the firework spec and task doc, and appended to the tags")
    s.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE", help="extra metadata (repeatable)")
    s.add_argument("--no-provenance", action="store_true", help="ignore a build-lewis provenance sidecar next to the structures file")
    s.add_argument("--fmax", type=float, default=0.01)
    s.add_argument("--max-steps", type=int, default=500)
    s.add_argument("--no-trajectory", action="store_true", help="do not store trajectories in the data store")
    s.add_argument("--dry-run", action="store_true", help="build workflows but do not write to the launchpad")
    s.set_defaults(func=cmd_load)

    s = sub.add_parser("status", help="monitor fireworks and task documents")
    s.add_argument("--launchpad", default=settings.DEFAULT_LAUNCHPAD)
    s.add_argument("--jobflow-config", default=settings.DEFAULT_JOBFLOW_CONFIG)
    s.add_argument("--category", default=settings.FW_CATEGORY)
    s.add_argument("--counts", action="store_true", help="FW states per batch (default)")
    s.add_argument("--doc", metavar="MOLECULE_ID", help="task doc summary from the JobStore")
    s.add_argument("--full", action="store_true", help="with --doc: dump the whole document (minus trajectory)")
    s.add_argument("--unconverged", action="store_true", help="docs whose optimizer hit max_steps")
    s.add_argument("--fizzled", action="store_true", help="FIZZLED fw_ids + last traceback line")
    s.add_argument("--run-label", help="with --doc/--unconverged: only task docs with this metadata.run_label")
    s.set_defaults(func=cmd_status)

    s = sub.add_parser("smoke-test", help="run_locally on a GPU node, no FireWorks")
    s.add_argument("structures")
    s.add_argument("--calc-config", default=settings.DEFAULT_CALC_CONFIG)
    s.add_argument("--outdir", help="default $CAMPAIGN_LAUNCH_ROOT/smoke/<jobid>")
    _add_selection(s)
    s.set_defaults(func=cmd_smoke_test)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
