"""Offline source-graph assembly, ownership, sharding, and resume checks."""

import json
import os

import pytest
from .fake_mmu import source_graph_kwargs, source_graph_stream

from mmu_stream.streaming import (
    MATCH_INDEX_ENV,
    _partition_owner,
    _source_graph_examples,
    load_match_index,
    open_stream,
    owned_by_rank,
    resolve_match_index,
    shuffled,
    split_files,
    split_of_cell,
)


def _ids(stream) -> list[str]:
    return [record["object_id"] for record in stream]


def test_source_graph_assembles_spokes_and_emits_only_eligible_unmatched(tmp_path):
    records = list(source_graph_stream(tmp_path))
    by_id = {record["object_id"]: record for record in records}

    assert len(records) == 10
    assert {key for key in by_id if ":" in key} == {
        "desi:d03",
        "desi:d13",
        "sdss:s02",
        "sdss:s12",
    }
    assert not any(key.startswith("provabgs:") for key in by_id)

    complete = by_id["a01"]
    assert {"image", "spectrum", "sdss_spectrum", "provabgs_LOG_MSTAR"} <= set(
        complete
    )
    assert "spectrum" in by_id["a02"] and "sdss_spectrum" not in by_id["a02"]
    assert "image" in by_id["a03"] and "spectrum" not in by_id["a03"]


def test_source_graph_resume_and_data_source_shards_are_exact(tmp_path):
    stream = source_graph_stream(tmp_path)
    assert stream.n_shards == 2
    iterator = iter(stream)
    for _ in range(3):
        next(iterator)
    state = stream.state_dict()
    expected = [next(iterator)["object_id"] for _ in range(4)]

    resumed = source_graph_stream(tmp_path)
    resumed.load_state_dict(state)
    assert [next(iter(resumed))["object_id"] for _ in range(1)] == expected[:1]
    resumed.load_state_dict(state)
    resumed_iterator = iter(resumed)
    assert [next(resumed_iterator)["object_id"] for _ in range(4)] == expected

    all_ids = set(_ids(source_graph_stream(tmp_path)))
    shards = [
        set(
            _ids(
                source_graph_stream(tmp_path).shard(
                    num_shards=2, index=index, contiguous=False
                )
            )
        )
        for index in range(2)
    ]
    assert shards[0] and shards[1]
    assert shards[0].isdisjoint(shards[1])
    assert shards[0] | shards[1] == all_ids


def test_missing_partner_warns_and_is_skipped(tmp_path):
    kwargs = source_graph_kwargs(tmp_path)
    matches = json.loads(kwargs["match_json"][0])
    matches["a01"]["sdss"] = "missing"
    kwargs["match_json"][0] = json.dumps(matches)
    kwargs["matched_source_ids"]["sdss"].append("missing")

    with pytest.warns(RuntimeWarning, match="missing sdss id 'missing'"):
        records = list(_source_graph_examples(**kwargs))
    assert "sdss_spectrum" not in next(r for r in records if r["object_id"] == "a01")


def test_split_shuffle_and_rank_ownership_are_deterministic():
    files = [f"f{i}" for i in range(165)]
    val = split_files(files, "val", val_partitions=8)
    train = split_files(files, "train", val_partitions=8)
    assert set(val).isdisjoint(train)
    assert sorted(val + train) == sorted(files)

    first = shuffled(files, seed=7, epoch=0)
    assert first == shuffled(files, seed=7, epoch=0)
    assert first != shuffled(files, seed=7, epoch=1)
    for ranks in (1, 2, 8, 64):
        deals = [owned_by_rank(first, rank, ranks) for rank in range(ranks)]
        assert sorted(path for deal in deals for path in deal) == sorted(files)


def test_common_spatial_split_and_partition_owner():
    for pixel in range(16):
        assert split_of_cell((6, pixel << 4)) == split_of_cell((4, pixel))
    references = [(6, 3), (6, 1), (6, 2)]
    owner = _partition_owner("path", references, [(6, 1), (6, 2)])
    assert owner in {(6, 1), (6, 2)}
    assert owner == _partition_owner(
        "path", list(reversed(references)), [(6, 1), (6, 2)]
    )
    assert _partition_owner("path", references, [(6, 9)]) is None


def test_legacy_match_index_round_trips(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = tmp_path / "index.parquet"
    pq.write_table(
        pa.table(
            {
                "image_order": pa.array([6, 6, 6], pa.int8()),
                "image_pixel": pa.array([7, 7, 9], pa.int64()),
                "image_id": ["i1", "i2", "i9"],
                "spectrum_order": pa.array([8, 8, 8], pa.int8()),
                "spectrum_pixel": pa.array([2, 3, 2], pa.int64()),
                "spectrum_id": ["s1", "s2", "s9"],
            }
        ),
        path,
    )
    matches, cells = load_match_index(str(path))
    assert matches == {(6, 7): {"i1": "s1", "i2": "s2"}, (6, 9): {"i9": "s9"}}
    assert cells == {(6, 7): {(8, 2), (8, 3)}, (6, 9): {(8, 2)}}


def test_match_index_resolution_prefers_explicit_argument(monkeypatch):
    monkeypatch.delenv(MATCH_INDEX_ENV, raising=False)
    assert resolve_match_index() is None
    monkeypatch.setenv(MATCH_INDEX_ENV, "/from-env.parquet")
    assert resolve_match_index() == "/from-env.parquet"
    assert resolve_match_index("/explicit.parquet") == "/explicit.parquet"


@pytest.mark.network
def test_live_stream_decodes_and_resumes():
    match_index = os.environ.get(MATCH_INDEX_ENV)
    if not match_index:
        pytest.skip(f"live test requires ${MATCH_INDEX_ENV}")
    stream = open_stream(seed=0, match_index=match_index)
    iterator = iter(stream)
    first = next(iterator)
    assert "object_id" in first and "image" in first
    state = stream.state_dict()
    expected = [next(iterator)["object_id"] for _ in range(3)]
    resumed = open_stream(seed=0, match_index=match_index)
    resumed.load_state_dict(state)
    resumed_iterator = iter(resumed)
    assert [next(resumed_iterator)["object_id"] for _ in range(3)] == expected
