"""Read legacy and hub-and-spoke match indexes through one small interface."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

Cell = tuple[int, int]


def as_number(value, kind, field: str):
    """Coerce one raw value to int/float, rejecting junk uniformly."""
    try:
        return kind(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid {field}: {value!r}") from error


@dataclass
class MatchGraph:
    """Pointer graph grouped in anchor stream order.

    ``matches[cell][anchor_id][source]`` is the partner id and
    ``partner_cells[cell][source]`` is the set of partner partitions that must
    be fetched while scanning that anchor cell.
    """

    schema_version: int
    anchor_source: str
    anchor_revision: str
    partner_revisions: dict[str, str]
    matches: dict[Cell, dict[str, dict[str, str]]]
    partner_cells: dict[Cell, dict[str, set[Cell]]]


def _legacy(table: dict) -> MatchGraph:
    matches: dict[Cell, dict[str, dict[str, str]]] = {}
    partner_cells: dict[Cell, dict[str, set[Cell]]] = {}
    for index, anchor_id in enumerate(table["image_id"]):
        anchor_cell = (
            as_number(table["image_order"][index], int, "image_order"),
            as_number(table["image_pixel"][index], int, "image_pixel"),
        )
        partner_cell = (
            as_number(table["spectrum_order"][index], int, "spectrum_order"),
            as_number(table["spectrum_pixel"][index], int, "spectrum_pixel"),
        )
        matches.setdefault(anchor_cell, {}).setdefault(str(anchor_id), {})["desi"] = (
            str(table["spectrum_id"][index])
        )
        partner_cells.setdefault(anchor_cell, {}).setdefault("desi", set()).add(
            partner_cell
        )
    return MatchGraph(1, "legacy_north", "", {"desi": ""}, matches, partner_cells)


def _v3(table: dict) -> MatchGraph:
    """One row per anchor object, a column block per spoke, null where absent.

    The rows arrive already grouped, so this only fans the blocks out — no
    per-edge regrouping, which is what made v2 cost ~47 us per edge on every
    rank at startup.
    """
    versions = {
        as_number(value, int, "index_schema_version")
        for value in table["index_schema_version"]
    }
    if versions != {3}:
        raise ValueError(f"match index mixes schema versions {sorted(versions)}")
    anchors = set(map(str, table["anchor_source"]))
    anchor_revisions = set(map(str, table["anchor_revision"]))
    if len(anchors) != 1 or len(anchor_revisions) != 1:
        raise ValueError("match index must have one pinned anchor source/revision")
    sources = sorted(
        name[: -len("_id")]
        for name in table
        if name.endswith("_id") and name != "anchor_id"
    )
    if not sources:
        raise ValueError("wide match index has no spoke columns")

    matches: dict[Cell, dict[str, dict[str, str]]] = {}
    partner_cells: dict[Cell, dict[str, set[Cell]]] = {}
    revisions: dict[str, str] = {}
    for index, anchor_id in enumerate(table["anchor_id"]):
        cell = (
            as_number(table["anchor_order"][index], int, "anchor_order"),
            as_number(table["anchor_pixel"][index], int, "anchor_pixel"),
        )
        spokes = matches.setdefault(cell, {}).setdefault(str(anchor_id), {})
        for source in sources:
            partner_id = table[f"{source}_id"][index]
            if partner_id is None:
                continue  # no match for this spoke; the span is simply absent
            separation = table[f"{source}_separation_arcsec"][index]
            radius = table[f"{source}_match_radius_arcsec"][index]
            try:
                separation_value, radius_value = float(separation), float(radius)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{source} match needs separation/radius") from error
            if (
                not math.isfinite(separation_value)
                or not math.isfinite(radius_value)
                or radius_value <= 0
                or not 0 <= separation_value <= radius_value
            ):
                raise ValueError(f"invalid {source} match for anchor {anchor_id!r}")
            revision = str(table[f"{source}_revision"][index])
            if revisions.setdefault(source, revision) != revision:
                raise ValueError(f"partner {source!r} mixes revisions")
            spokes[source] = str(partner_id)
            partner_cells.setdefault(cell, {}).setdefault(source, set()).add(
                (
                    as_number(table[f"{source}_order"][index], int, f"{source}_order"),
                    as_number(table[f"{source}_pixel"][index], int, f"{source}_pixel"),
                )
            )
    return MatchGraph(
        3,
        anchors.pop(),
        anchor_revisions.pop(),
        revisions,
        matches,
        partner_cells,
    )


def _v2(table: dict) -> MatchGraph:
    versions = {
        as_number(value, int, "index_schema_version")
        for value in table["index_schema_version"]
    }
    if versions != {2}:
        raise ValueError(f"match index mixes schema versions {sorted(versions)}")
    anchors = set(map(str, table["anchor_source"]))
    anchor_revisions = set(map(str, table["anchor_revision"]))
    if len(anchors) != 1 or len(anchor_revisions) != 1:
        raise ValueError("match index must have one pinned anchor source/revision")

    matches: dict[Cell, dict[str, dict[str, str]]] = {}
    partner_cells: dict[Cell, dict[str, set[Cell]]] = {}
    revisions: dict[str, str] = {}
    for index, anchor_id in enumerate(table["anchor_id"]):
        # every spoke is an lsdb positional crossmatch to the anchor; there is
        # no identifier/lineage join to validate a second row shape for
        join_kind = str(table["join_kind"][index])
        if join_kind != "positional":
            raise ValueError(f"unknown join kind {join_kind!r}")
        try:
            separation_value = float(table["separation_arcsec"][index])
            radius_value = float(table["match_radius_arcsec"][index])
        except (TypeError, ValueError) as error:
            raise ValueError(
                "positional index rows require separation and radius"
            ) from error
        if (
            not math.isfinite(separation_value)
            or not math.isfinite(radius_value)
            or radius_value <= 0
            or not 0 <= separation_value <= radius_value
        ):
            raise ValueError("invalid positional match-index row")

        source = str(table["partner_source"][index])
        revision = str(table["partner_revision"][index])
        if source in revisions and revisions[source] != revision:
            raise ValueError(f"partner {source!r} mixes revisions")
        revisions[source] = revision
        anchor_cell = (
            as_number(table["anchor_order"][index], int, "anchor_order"),
            as_number(table["anchor_pixel"][index], int, "anchor_pixel"),
        )
        partner_cell = (
            as_number(table["partner_order"][index], int, "partner_order"),
            as_number(table["partner_pixel"][index], int, "partner_pixel"),
        )
        partners = matches.setdefault(anchor_cell, {}).setdefault(str(anchor_id), {})
        partner_id = str(table["partner_id"][index])
        if source in partners and partners[source] != partner_id:
            raise ValueError(f"anchor {anchor_id!r} has multiple {source!r} partners")
        partners[source] = partner_id
        partner_cells.setdefault(anchor_cell, {}).setdefault(source, set()).add(
            partner_cell
        )
    return MatchGraph(
        2,
        anchors.pop(),
        anchor_revisions.pop(),
        revisions,
        matches,
        partner_cells,
    )


def load_source_graph(path: str | Path) -> MatchGraph:
    """Load one parquet file or a directory of same-schema spoke parquets.

    Only ``*.parquet`` is read: an index directory usually collects a build
    log or a README alongside the spokes, and handing those to pyarrow fails
    the whole load at train start.
    """
    local = Path(path)
    source: str | Path | list[Path] = path
    if local.is_dir():
        source = sorted(local.glob("*.parquet"))
        if not source:
            raise ValueError(f"no spoke parquets in {path}")
    table = pq.read_table(source).to_pydict()
    names = set(table)
    if {
        "image_order",
        "image_pixel",
        "image_id",
        "spectrum_order",
        "spectrum_pixel",
        "spectrum_id",
    } <= names:
        return _legacy(table)
    if "partner_source" not in names:
        return _v3(table)
    required = {
        "index_schema_version",
        "anchor_source",
        "anchor_revision",
        "anchor_order",
        "anchor_pixel",
        "anchor_id",
        "partner_source",
        "partner_revision",
        "partner_order",
        "partner_pixel",
        "partner_id",
        "join_kind",
        "separation_arcsec",
        "match_radius_arcsec",
        "epoch_treatment",
    }
    missing = required - names
    if missing:
        raise ValueError(f"match index is missing columns {sorted(missing)}")
    return _v2(table)
