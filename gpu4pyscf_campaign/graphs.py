"""Bond-graph helpers: the intended graph stored in an entry vs the graph
OpenBabelNN perceives from coordinates (same recipe as crn_utils.utils and the
Lewis solver's acceptance test)."""

from __future__ import annotations

import logging

from pymatgen.core.structure import Molecule

logger = logging.getLogger(__name__)


def _edge_set(graph) -> frozenset[tuple[int, int]]:
    return frozenset((min(u, v), max(u, v)) for u, v in graph.edges())


def edges_from_graph_dict(mg_dict: dict) -> frozenset[tuple[int, int]]:
    """Edge set of a serialized pymatgen MoleculeGraph (an entry's ``molecule_graph``)."""
    from pymatgen.analysis.graphs import MoleculeGraph  # noqa: PLC0415

    return _edge_set(MoleculeGraph.from_dict(mg_dict).graph)


def intended_edges(entry: dict) -> frozenset[tuple[int, int]] | None:
    """The bond graph an entry is *meant* to have; None if the entry has no graph."""
    mg = entry.get("molecule_graph")
    return edges_from_graph_dict(mg) if mg else None


def perceive_edges(mol: Molecule) -> frozenset[tuple[int, int]] | None:
    """Bond graph OpenBabelNN reads from the coordinates; None if OpenBabel
    fails on this geometry."""
    from pymatgen.analysis.graphs import MoleculeGraph  # noqa: PLC0415
    from pymatgen.analysis.local_env import OpenBabelNN  # noqa: PLC0415

    try:
        mg = MoleculeGraph.from_local_env_strategy(mol, OpenBabelNN())
    except Exception:  # noqa: BLE001 -- openbabel raises assorted errors on bad geometries
        logger.debug("OpenBabelNN failed", exc_info=True)
        return None
    return _edge_set(mg.graph)
