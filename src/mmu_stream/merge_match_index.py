"""Pivot a directory of spoke indexes into one row per anchor object.

The spoke files are an edge list: one row per (anchor, spoke) match, so an
anchor with three spokes occupies three rows and the loader has to regroup
them in Python on every rank. This writes the grouped form instead — one row
per anchor object, a column block per spoke, null where that spoke has no
match, which is exactly the span the sequencer then skips.

    uv run --extra index python -m mmu_stream.merge_match_index \
      --spokes .../north-v2 --out .../north-v2-merged/match_index.parquet

Sources are discovered from the spoke files; the anchor and every spoke
revision are carried through so the merged file validates the same pins.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

WIDE_SCHEMA_VERSION = 3
ANCHOR_FIELDS = [
    ("index_schema_version", pa.int16()),
    ("anchor_source", pa.string()),
    ("anchor_revision", pa.string()),
    ("anchor_order", pa.int8()),
    ("anchor_pixel", pa.int64()),
    ("anchor_id", pa.string()),
]
SPOKE_FIELDS = [
    ("revision", pa.string()),
    ("id", pa.string()),
    ("order", pa.int8()),
    ("pixel", pa.int64()),
    ("separation_arcsec", pa.float32()),
    ("match_radius_arcsec", pa.float32()),
]


def wide_schema(sources) -> pa.Schema:
    fields = list(ANCHOR_FIELDS)
    for source in sources:
        fields += [(f"{source}_{name}", kind) for name, kind in SPOKE_FIELDS]
    fields.append(("epoch_treatment", pa.string()))
    return pa.schema(fields)


def _number(value, field: str, kind):
    try:
        return kind(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"invalid {field}: {value!r}") from error


def pivot(table: pa.Table) -> pa.Table:
    """Edge rows -> one row per anchor, a null block per absent spoke."""
    edges = table.to_pydict()
    sources = sorted(set(edges["partner_source"]))
    anchors: dict = {}
    anchor_source, anchor_revision, epoch = set(), set(), set()
    for index, anchor_id in enumerate(edges["anchor_id"]):
        anchor_source.add(str(edges["anchor_source"][index]))
        anchor_revision.add(str(edges["anchor_revision"][index]))
        epoch.add(str(edges["epoch_treatment"][index]))
        key = (
            _number(edges["anchor_order"][index], "anchor_order", int),
            _number(edges["anchor_pixel"][index], "anchor_pixel", int),
            str(anchor_id),
        )
        source = str(edges["partner_source"][index])
        spoke = {
            "revision": str(edges["partner_revision"][index]),
            "id": str(edges["partner_id"][index]),
            "order": _number(edges["partner_order"][index], "partner_order", int),
            "pixel": _number(edges["partner_pixel"][index], "partner_pixel", int),
            "separation_arcsec": _number(
                edges["separation_arcsec"][index], "separation_arcsec", float
            ),
            "match_radius_arcsec": _number(
                edges["match_radius_arcsec"][index], "match_radius_arcsec", float
            ),
        }
        present = anchors.setdefault(key, {})
        if present.setdefault(source, spoke) != spoke:
            raise ValueError(f"anchor {anchor_id!r} has two {source!r} matches")
    if len(anchor_source) != 1 or len(anchor_revision) != 1 or len(epoch) != 1:
        raise ValueError("spokes disagree on anchor source, revision, or epoch")

    schema = wide_schema(sources)
    rows: dict = {name: [] for name in schema.names}
    for (order, pixel, anchor_id), spokes in sorted(anchors.items()):
        rows["index_schema_version"].append(WIDE_SCHEMA_VERSION)
        rows["anchor_source"].append(next(iter(anchor_source)))
        rows["anchor_revision"].append(next(iter(anchor_revision)))
        rows["anchor_order"].append(order)
        rows["anchor_pixel"].append(pixel)
        rows["anchor_id"].append(anchor_id)
        rows["epoch_treatment"].append(next(iter(epoch)))
        for source in sources:
            spoke = spokes.get(source)
            for name, _ in SPOKE_FIELDS:
                rows[f"{source}_{name}"].append(None if spoke is None else spoke[name])
    return pa.table(rows, schema=schema)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spokes", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    parts = sorted(args.spokes.glob("*.parquet"))
    if not parts:
        raise ValueError(f"no spoke parquets in {args.spokes}")
    merged = pivot(pa.concat_tables([pq.read_table(part) for part in parts]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(merged, args.out)
    print(f"wrote {merged.num_rows} anchors -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
