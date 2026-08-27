"""Offline checks for the shared ADR 0013 match-index builder."""

from types import SimpleNamespace

import pytest

import mmu_stream.build_match_index as builder
import mmu_stream.merge_match_index as merger
from mmu_stream.match_index import load_source_graph
from mmu_stream.streaming import SOURCE_GRAPH_ASSEMBLY, source_assembly_for_index


def test_index_schema_and_row_building():
    assert builder.normalize_id(b"    42", strip=True) == "42"
    assert builder.normalize_id("b'    42'", strip=True) == "42"
    assert builder.SCHEMA.names == [
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
    ]

    import argparse

    import pandas as pd

    # one lsdb crossmatch partition: left columns bare, right suffixed
    pairs = pd.DataFrame(
        {
            "object_id": ["anchor-1"],
            "ra": [10.0],
            "dec": [20.0],
            f"dr8_id{builder.PARTNER_SUFFIX}": [b"  7 "],
            f"ra{builder.PARTNER_SUFFIX}": [10.0001],
            f"dec{builder.PARTNER_SUFFIX}": [20.0],
            builder.DISTANCE_COLUMN: [0.34],
        }
    )
    args = argparse.Namespace(
        anchor_source="legacy_north",
        anchor_revision="north-sha",
        anchor_id_column="object_id",
        anchor_strip_id=False,
        partner_source="galaxies_train",
        partner_revision="partner-sha",
        partner_id_column="dr8_id",
        partner_strip_id=True,
        radius_arcsec=1.0,
        epoch_treatment="icrs_j2000_static",
    )
    # lsdb's alignment supplies both cells, keyed by crossmatch partition
    anchor_cell, partner_cell = (6, 19863), (5, 4965)
    pixel = SimpleNamespace(order=8, pixel=79452)
    cells = {(pixel.order, pixel.pixel): (anchor_cell, partner_cell)}
    rows = builder._rows(pairs, pixel, args, cells)

    assert rows["anchor_id"].tolist() == ["anchor-1"]
    assert rows["partner_id"].tolist() == ["7"]
    assert rows["anchor_order"].tolist() == [6]
    assert rows["anchor_pixel"].tolist() == [anchor_cell[1]]
    assert rows["partner_order"].tolist() == [5]
    assert rows["partner_pixel"].tolist() == [partner_cell[1]]
    assert rows["join_kind"].tolist() == ["positional"]
    assert rows["separation_arcsec"].tolist() == pytest.approx([0.34])

    # an empty partition, and the (0, 0) probe lsdb builds meta with, must both
    # match the real output's dtypes or dask rejects the partition
    for empty in (
        builder._rows(pairs.iloc[:0], pixel, args, cells),
        builder._rows(pairs, SimpleNamespace(order=0, pixel=0), args, cells),
    ):
        assert list(empty.columns) == list(builder.EMPTY_ROWS.columns)
        assert empty.dtypes.equals(rows.dtypes)


def test_v2_spoke_directory_loads_as_one_source_graph(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = tmp_path / "indexes"
    directory.mkdir()
    common = {
        "index_schema_version": 2,
        "anchor_source": "legacy_north",
        "anchor_revision": "north-sha",
        "anchor_order": 6,
        "anchor_pixel": 7,
        "anchor_id": "anchor-1",
        "partner_revision": "partner-sha",
        "partner_order": 8,
        "partner_pixel": 9,
        "separation_arcsec": 0.2,
        "match_radius_arcsec": 1.0,
        "epoch_treatment": "icrs_j2000_static",
        "join_kind": "positional",
    }
    for source, partner_id in (("desi", "spectrum-1"), ("galaxies", "galaxy-1")):
        row = {**common, "partner_source": source, "partner_id": partner_id}
        pq.write_table(
            pa.Table.from_pylist([row], schema=builder.SCHEMA),
            directory / f"{source}.parquet",
        )

    graph = load_source_graph(directory)
    assert graph.schema_version == 2
    assert graph.anchor_source == "legacy_north"
    assert graph.matches == {
        (6, 7): {"anchor-1": {"desi": "spectrum-1", "galaxies": "galaxy-1"}}
    }
    assert graph.partner_cells == {(6, 7): {"desi": {(8, 9)}, "galaxies": {(8, 9)}}}
    assert source_assembly_for_index(str(directory)) == SOURCE_GRAPH_ASSEMBLY

    # every spoke is positional now, so any other join kind is rejected
    bad = {
        **common,
        "partner_source": "desi",
        "partner_id": "x",
        "join_kind": "lineage",
    }
    pq.write_table(
        pa.Table.from_pylist([bad], schema=builder.SCHEMA), directory / "bad.parquet"
    )
    with pytest.raises(ValueError, match="unknown join kind"):
        load_source_graph(directory)
    (directory / "bad.parquet").unlink()

    invalid = {**common, "partner_source": "sdss", "partner_id": "x", "separation_arcsec": 2.0}
    pq.write_table(
        pa.Table.from_pylist([invalid], schema=builder.SCHEMA),
        directory / "invalid.parquet",
    )
    with pytest.raises(ValueError, match="invalid positional"):
        load_source_graph(directory)
    (directory / "invalid.parquet").unlink()

    mixed = {
        **common,
        "anchor_id": "anchor-2",
        "partner_source": "desi",
        "partner_revision": "other-sha",
        "partner_id": "x",
    }
    pq.write_table(
        pa.Table.from_pylist([mixed], schema=builder.SCHEMA),
        directory / "mixed.parquet",
    )
    with pytest.raises(ValueError, match="mixes revisions"):
        load_source_graph(directory)


def test_wide_index_matches_the_edge_list(tmp_path):
    """The one-row-per-anchor pivot must load to the identical graph."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    common = {
        "index_schema_version": 2,
        "anchor_source": "legacy_north",
        "anchor_revision": "north-sha",
        "anchor_order": 6,
        "match_radius_arcsec": 1.0,
        "epoch_treatment": "icrs_j2000_static",
        "join_kind": "positional",
        "separation_arcsec": 0.2,
        "partner_order": 8,
        "partner_pixel": 9,
    }
    edges = [
        # anchor-1 has both spokes, anchor-2 only desi, anchor-3 only hsc
        {
            **common,
            "anchor_pixel": 7,
            "anchor_id": "anchor-1",
            "partner_source": "desi",
            "partner_revision": "d",
            "partner_id": "s1",
        },
        {
            **common,
            "anchor_pixel": 7,
            "anchor_id": "anchor-1",
            "partner_source": "hsc",
            "partner_revision": "h",
            "partner_id": "i1",
        },
        {
            **common,
            "anchor_pixel": 7,
            "anchor_id": "anchor-2",
            "partner_source": "desi",
            "partner_revision": "d",
            "partner_id": "s2",
        },
        {
            **common,
            "anchor_pixel": 9,
            "anchor_id": "anchor-3",
            "partner_source": "hsc",
            "partner_revision": "h",
            "partner_id": "i3",
        },
    ]
    spokes = tmp_path / "spokes"
    spokes.mkdir()
    for source in ("desi", "hsc"):
        rows = [e for e in edges if e["partner_source"] == source]
        pq.write_table(
            pa.Table.from_pylist(rows, schema=builder.SCHEMA),
            spokes / f"{source}.parquet",
        )

    wide = merger.pivot(pq.read_table(spokes))
    wide_dir = tmp_path / "merged"
    wide_dir.mkdir()
    pq.write_table(wide, wide_dir / "match_index.parquet")

    assert wide.num_rows == 3  # one row per anchor, not per edge
    columns = wide.to_pydict()
    assert columns["desi_id"] == ["s1", "s2", None]
    assert columns["hsc_id"] == ["i1", None, "i3"]  # absent spoke is null

    edge_graph = load_source_graph(spokes)
    wide_graph = load_source_graph(wide_dir)
    assert wide_graph.schema_version == 3

    # the layout must not change which streaming assembly is selected, or the
    # stream falls back to the DESI-only path and rejects the other spokes
    assert source_assembly_for_index(str(spokes)) == SOURCE_GRAPH_ASSEMBLY
    assert source_assembly_for_index(str(wide_dir)) == SOURCE_GRAPH_ASSEMBLY
    assert wide_graph.matches == edge_graph.matches
    assert wide_graph.partner_cells == edge_graph.partner_cells
    assert wide_graph.partner_revisions == edge_graph.partner_revisions
    assert wide_graph.anchor_revision == edge_graph.anchor_revision

    # a spoke that disagrees with itself must not be silently merged away
    conflict = {**edges[0], "partner_id": "s-other"}
    pq.write_table(
        pa.Table.from_pylist(edges + [conflict], schema=builder.SCHEMA),
        spokes / "desi.parquet",
    )
    with pytest.raises(ValueError, match="two 'desi' matches"):
        merger.pivot(pq.read_table(spokes))
