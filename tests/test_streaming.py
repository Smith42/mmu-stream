"""Source-graph ownership, index compatibility, and live stream checks."""

import os

import pytest

from mmu_stream.streaming import (
    MATCH_INDEX_ENV,
    _partition_owner,
    load_match_index,
    open_stream,
    owned_by_rank,
    resolve_match_index,
    shuffled,
    split_files,
    split_of_cell,
)


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
