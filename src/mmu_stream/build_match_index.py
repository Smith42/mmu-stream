"""Build one pointer-only spoke index off a single HATS hub.

LSDB is confined to this offline builder, and the positional join is LSDB's
own default: ``anchor.crossmatch(spoke)`` — KdTree, one neighbour, 1 arcsec.
One anchor, one spoke per run; spokes never match each other.

The MMU catalogs are HATS collections with a 10-arcsec ``default_margin``,
which lsdb attaches on open, so those joins are exact at partition edges.
galaxies-with-hats publishes no margin and is lossy there; each build prints
which of the two it got.

Pinned North × galaxies-with-hats train example::

    uv run --extra index python -m mmu_stream.build_match_index \
      --anchor-catalog hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north@f634744d3c44dd4fde0dee3172d4887c5e3c31c0 \
      --anchor-source legacy_north \
      --anchor-revision f634744d3c44dd4fde0dee3172d4887c5e3c31c0 \
      --partner-catalog hf://datasets/Smith42/galaxies-with-hats@c0188b776c4ce6312a805a04cbc25c891a075933/train \
      --partner-source galaxies_train \
      --partner-revision c0188b776c4ce6312a805a04cbc25c891a075933 \
      --partner-id-column dr8_id \
      --out galaxies_train.parquet

Positional builds are one dask graph over the crossmatch partitions, computed
on ``--workers`` threads. The output stores source names, revisions, both HATS
cells/ids, separation, radius, epoch treatment, and schema version. A directory
of same-schema spoke parquets is one cumulative index consumable by
``load_source_graph``.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import time
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .match_index import as_number

INDEX_SCHEMA_VERSION = 2
# lsdb suffixes the right side; the left keeps its own column names
PARTNER_SUFFIX = "_spoke"
DISTANCE_COLUMN = "_dist_arcsec"
SCHEMA = pa.schema(
    [
        ("index_schema_version", pa.int16()),
        ("anchor_source", pa.string()),
        ("anchor_revision", pa.string()),
        ("anchor_order", pa.int8()),
        ("anchor_pixel", pa.int64()),
        ("anchor_id", pa.string()),
        ("partner_source", pa.string()),
        ("partner_revision", pa.string()),
        ("partner_order", pa.int8()),
        ("partner_pixel", pa.int64()),
        ("partner_id", pa.string()),
        ("join_kind", pa.string()),
        ("separation_arcsec", pa.float32()),
        ("match_radius_arcsec", pa.float32()),
        ("epoch_treatment", pa.string()),
    ]
)


# arrow-backed dtypes so an all-null via_* column stays int8/int64 instead of
# degrading to float64 and mismatching the dask meta
EMPTY_ROWS = SCHEMA.empty_table().to_pandas(types_mapper=pd.ArrowDtype)


def normalize_id(value: Any, strip: bool = False) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value)
    if strip and text.startswith("b'"):
        try:
            decoded = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            decoded = text
        if isinstance(decoded, bytes):
            text = decoded.decode("utf-8")
    return text.strip() if strip else text


def alignment_cells(anchor, partner) -> dict:
    """Crossmatch partition -> (anchor cell, partner cell), from lsdb itself.

    ``align_catalogs`` is the helper ``crossmatch`` calls to lay out its own
    partitions — margin tree and MOC filtering included — so its mapping is
    the definition of which cell of each side fed each partition. Deriving it
    from row coordinates instead only invents ways to disagree with lsdb.
    """
    align_catalogs = cast(
        Any, importlib.import_module("lsdb.dask.merge_catalog_functions")
    ).align_catalogs

    mapping = align_catalogs(anchor, partner).pixel_mapping
    cells: dict = {}
    for row in mapping.itertuples(index=False):
        aligned = (
            as_number(row.aligned_Norder, int, "aligned_Norder"),
            as_number(row.aligned_Npix, int, "aligned_Npix"),
        )
        sides = (
            (
                as_number(row.primary_Norder, int, "primary_Norder"),
                as_number(row.primary_Npix, int, "primary_Npix"),
            ),
            (
                as_number(row.join_Norder, int, "join_Norder"),
                as_number(row.join_Npix, int, "join_Npix"),
            ),
        )
        if cells.setdefault(aligned, sides) != sides:
            raise ValueError(f"crossmatch partition {aligned} maps to two cells")
    return cells


def _rows(pairs, pixel, args, cells):
    """One crossmatch partition -> pointer rows. Runs as a dask task.

    ``pixel`` is the crossmatch partition, handed over by lsdb's own
    ``map_partitions(include_pixel=True)``.
    """
    rows = {name: [] for name in SCHEMA.names}
    key = (
        as_number(pixel.order, int, "pixel.order"),
        as_number(pixel.pixel, int, "pixel.pixel"),
    )
    if len(pairs) == 0 or key not in cells:
        return EMPTY_ROWS
    if DISTANCE_COLUMN not in pairs:
        raise ValueError(f"crossmatch result has no {DISTANCE_COLUMN} column")
    anchor_cell, partner_cell = cells[key]
    for position in range(len(pairs)):
        row = pairs.iloc[position]
        values = {
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "anchor_source": args.anchor_source,
            "anchor_revision": args.anchor_revision,
            "anchor_order": anchor_cell[0],
            "anchor_pixel": anchor_cell[1],
            "anchor_id": normalize_id(row[args.anchor_id_column], args.anchor_strip_id),
            "partner_source": args.partner_source,
            "partner_revision": args.partner_revision,
            "partner_order": partner_cell[0],
            "partner_pixel": partner_cell[1],
            "partner_id": normalize_id(
                row[f"{args.partner_id_column}{PARTNER_SUFFIX}"], args.partner_strip_id
            ),
            "join_kind": "positional",
            "separation_arcsec": as_number(
                row[DISTANCE_COLUMN], float, DISTANCE_COLUMN
            ),
            "match_radius_arcsec": args.radius_arcsec,
            "epoch_treatment": args.epoch_treatment,
        }
        for name, value in values.items():
            rows[name].append(value)
    # via pyarrow so empty and full partitions agree on dtypes, and both agree
    # with the dask meta — otherwise dask rejects the partition
    return pa.table(rows, schema=SCHEMA).to_pandas(types_mapper=pd.ArrowDtype)


def parse_cell(value: str) -> tuple[int, int]:
    try:
        order, pixel = value.split("/", 1)
        return (
            as_number(order, int, "cell order"),
            as_number(pixel, int, "cell pixel"),
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError("cell must be ORDER/PIXEL") from error


def build(args) -> int:
    if args.radius_arcsec <= 0:
        raise ValueError("radius_arcsec must be positive")
    if not args.anchor_revision or not args.partner_revision:
        raise ValueError("source revisions must be pinned")

    dask = cast(Any, importlib.import_module("dask"))
    lsdb = cast(Any, importlib.import_module("lsdb"))
    anchor = lsdb.open_catalog(
        args.anchor_catalog,
        columns=[args.anchor_id_column, "ra", "dec"],
    )
    partner = lsdb.open_catalog(
        args.partner_catalog,
        columns=[args.partner_id_column, "ra", "dec"],
    )
    # the MMU collections carry a 10-arcsec default margin that lsdb attaches
    # here; a spoke without one (galaxies-with-hats) silently loses pairs whose
    # partner sits across a partition edge, so say which one this build got
    margin = getattr(partner, "margin", None)
    print(
        f"{args.partner_source}: margin cache "
        + (
            f"{margin.hc_structure.catalog_info.margin_threshold} arcsec"
            if margin is not None
            else "MISSING — matches are lossy at partition edges"
        ),
        flush=True,
    )

    cells = alignment_cells(anchor, partner)
    # bound a smoke build by narrowing the anchor with lsdb's own spatial
    # filter — slicing the crossmatch's dask partitions afterwards instead
    # produces a graph with missing dependencies. Pick the cells from the
    # alignment, since anchor cells in HEALPix order need not meet the spoke
    # at all and lsdb refuses a non-overlapping pair outright.
    overlapping = list(dict.fromkeys(anchor_cell for anchor_cell, _ in cells.values()))
    if args.cell is not None:
        if args.cell not in overlapping:
            raise ValueError(f"anchor cell {args.cell} does not meet the spoke")
        overlapping = [args.cell]
    elif args.limit_partitions is not None:
        overlapping = overlapping[: args.limit_partitions]
    if len(overlapping) < len(cells):
        anchor = anchor.pixel_search(overlapping, fine=False)
        cells = alignment_cells(anchor, partner)

    matched = anchor.crossmatch(
        partner,
        radius_arcsec=args.radius_arcsec,
        suffixes=("", PARTNER_SUFFIX),
        # pinned: lsdb's default flips to "overlapping_columns" in a future
        # release, and the suffix is what identifies the spoke side here
        suffix_method="all_columns",
    )
    pixels = matched.get_ordered_healpix_pixels()
    unknown = [p for p in pixels if (p.order, p.pixel) not in cells]
    if unknown:
        raise ValueError(
            f"{len(unknown)} crossmatch partitions are absent from the pixel "
            f"alignment (first: {unknown[0]})"
        )

    # lsdb's own map_partitions hands each task its HealpixPixel, and each task
    # is one HTTP-bound read, so dask's thread pool is the whole parallelism
    # story. The result is pointers — the DESI index is ~0.7M rows — so it is
    # collected and written once rather than streamed out in parts.
    rows = matched.map_partitions(
        _rows, args, cells, meta=EMPTY_ROWS, include_pixel=True
    )
    print(
        f"{args.partner_source}: {len(pixels)} crossmatch partitions "
        f"-> {args.out} on {args.workers} threads",
        flush=True,
    )
    started = time.time()
    with dask.config.set(scheduler="threads", num_workers=args.workers):
        frame = rows.compute()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, schema=SCHEMA, preserve_index=False)
    pq.write_table(table, args.out)
    print(
        f"{args.partner_source}: {table.num_rows} matches "
        f"in {time.time() - started:.0f}s"
    )
    return table.num_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-catalog", required=True)
    parser.add_argument("--anchor-source", required=True)
    parser.add_argument("--anchor-revision", required=True)
    parser.add_argument("--anchor-id-column", default="object_id")
    parser.add_argument("--anchor-strip-id", action="store_true")
    parser.add_argument("--partner-catalog", required=True)
    parser.add_argument("--partner-source", required=True)
    parser.add_argument("--partner-revision", required=True)
    parser.add_argument("--partner-id-column", default="object_id")
    parser.add_argument("--partner-strip-id", action="store_true")
    parser.add_argument("--radius-arcsec", type=float, default=1.0)
    parser.add_argument("--epoch-treatment", default="icrs_j2000_static")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--cell", type=parse_cell)
    selection.add_argument("--limit-partitions", type=int)
    # each task is one small HTTP read, so oversubscribe cores; too many at once
    # gets SSL read errors back from the hub
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    count = build(args)
    print(f"wrote {count} matches -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
