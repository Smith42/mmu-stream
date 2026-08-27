"""Deterministic local hub-and-spoke parquet fixture."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mmu_stream.streaming import source_graph_dataset


def _write(path: Path, rows: list[dict]) -> str:
    pq.write_table(pa.Table.from_pylist(rows), path)
    return str(path)


def _image() -> dict:
    return {
        "flux": np.zeros((3, 152, 152), dtype=np.float32).tolist(),
        "band": ["des-g", "des-r", "des-z"],
        "psf_fwhm": [1.2, 1.1, 1.0],
    }


def _spectrum() -> dict:
    return {
        "flux": [1.0, 2.0],
        "lambda": [4000.0, 4001.0],
        "mask": [False, False],
    }


def source_graph_kwargs(root: Path) -> dict:
    """Write two source-graph cells and return generator arguments."""
    root.mkdir(parents=True, exist_ok=True)
    image_paths, match_json, partition_json, owned_paths = [], [], [], []
    matched = {"desi": [], "sdss": [], "provabgs": []}

    import json

    for cell in range(2):
        anchors = []
        for number in range(1, 4):
            anchors.append(
                {
                    "object_id": f"a{cell}{number}",
                    "ra": 10.0 + cell,
                    "dec": 20.0,
                    "_healpix_29": cell * 10 + number,
                    "image": _image(),
                    "ebv": 0.01,
                    "flux_g": 1.0,
                    "flux_r": 1.0,
                    "flux_z": 1.0,
                    "z_spec": 0.2,
                    "fiberflux_g": 0.5,
                    "fiberflux_r": 0.5,
                    "fiberflux_z": 0.5,
                    "psfdepth_g": 100.0,
                    "psfdepth_r": 100.0,
                    "psfdepth_z": 100.0,
                }
            )
        image_paths.append(_write(root / f"anchors-{cell}.parquet", anchors))

        desi = [
            {
                "object_id": f"d{cell}{number}",
                "ra": 10.0 + cell,
                "dec": 20.0,
                "_healpix_29": cell * 10 + number,
                "spectrum": _spectrum(),
                "Z": 0.2,
                "ZERR": 0.01,
                "ZWARN": False,
            }
            for number in range(1, 4)
        ]
        sdss = [
            {
                "object_id": f"s{cell}{number}".encode(),
                "ra": 10.0 + cell,
                "dec": 20.0,
                "_healpix_29": cell * 10 + number,
                "spectrum": _spectrum(),
                "Z": 0.3,
                "Z_ERR": 0.02,
                "ZWARNING": False,
            }
            for number in range(1, 3)
        ]
        provabgs = [
            {
                "object_id": f"p{cell}{number}",
                "ra": 10.0 + cell,
                "dec": 20.0,
                "_healpix_29": cell * 10 + number,
                "LOG_MSTAR": 10.0,
                "Z_HP": 0.2,
                "Z_MW": 0.01,
                "TAGE_MW": 8.0,
                "AVG_SFR": 1.0,
                "TSNR2_BGS": 100.0,
            }
            for number in range(1, 3)
        ]
        paths = {
            "desi": _write(root / f"desi-{cell}.parquet", desi),
            "sdss": _write(root / f"sdss-{cell}.parquet", sdss),
            "provabgs": _write(root / f"provabgs-{cell}.parquet", provabgs),
        }
        matches = {
            f"a{cell}1": {
                "desi": f"d{cell}1",
                "sdss": f"s{cell}1",
                "provabgs": f"p{cell}1",
            },
            f"a{cell}2": {"desi": f"d{cell}2"},
        }
        for source_ids in matches.values():
            for source, source_id in source_ids.items():
                matched[source].append(source_id)
        match_json.append(json.dumps(matches))
        partition_json.append(
            json.dumps(
                {
                    source: [{"path": path, "order": 6, "pixel": cell}]
                    for source, path in paths.items()
                }
            )
        )
        owned_paths.append({source: [path] for source, path in paths.items()})

    return {
        "image_paths": image_paths,
        "match_json": match_json,
        "source_partitions_json": partition_json,
        "owned_source_paths": owned_paths,
        "matched_source_ids": matched,
    }


def source_graph_stream(root: Path):
    return source_graph_dataset(**source_graph_kwargs(root))
