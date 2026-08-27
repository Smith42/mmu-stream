"""Stream the crossmatch-only MMU corpus natively at train time.

The precomputed match index (``scripts/build_match_index.py``) defines the
corpus. One pass over its LegacySurvey partitions emits every downloaded row:
matched image × spectrum pairs, unmatched images, and globally unmatched
spectra. There is no standalone source, weighting, skim governor, or local
cache; the modality mix follows the data.

``datasets.IterableDataset.from_generator`` owns worker sharding and resume.
DP ranks are split by dealing the partition list (``owned_by_rank``), not by
``split_dataset_by_node``. Never call ``IterableDataset.shuffle()``: datasets
5.x collapses ``n_shards`` to one. Instead, partition paths are
deterministically permuted per epoch.
"""

from __future__ import annotations

import ast
import importlib
import json
import math
import os
import warnings
import zlib
from typing import Any, cast

import numpy as np

from .match_index import load_source_graph

GWH_FRACTION_FIELDS = (
    "smooth-or-featured_smooth_fraction",
    "smooth-or-featured_featured-or-disk_fraction",
    "smooth-or-featured_artifact_fraction",
    "disk-edge-on_yes_fraction",
    "disk-edge-on_no_fraction",
    "has-spiral-arms_yes_fraction",
    "has-spiral-arms_no_fraction",
    "bar_strong_fraction",
    "bar_weak_fraction",
    "bar_no_fraction",
    "bulge-size_dominant_fraction",
    "bulge-size_large_fraction",
    "bulge-size_moderate_fraction",
    "bulge-size_small_fraction",
    "bulge-size_none_fraction",
    "how-rounded_round_fraction",
    "how-rounded_in-between_fraction",
    "how-rounded_cigar-shaped_fraction",
    "edge-on-bulge_boxy_fraction",
    "edge-on-bulge_none_fraction",
    "edge-on-bulge_rounded_fraction",
    "spiral-winding_tight_fraction",
    "spiral-winding_medium_fraction",
    "spiral-winding_loose_fraction",
    "spiral-arm-count_1_fraction",
    "spiral-arm-count_2_fraction",
    "spiral-arm-count_3_fraction",
    "spiral-arm-count_4_fraction",
    "spiral-arm-count_more-than-4_fraction",
    "spiral-arm-count_cant-tell_fraction",
    "merging_none_fraction",
    "merging_minor-disturbance_fraction",
    "merging_major-disturbance_fraction",
    "merging_merger_fraction",
)

IMAGES_CATALOG = "hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north"
SPECTRA_CATALOG = "hf://datasets/UniverseTBD/mmu_desi_edr_sv3"
SOURCE_CATALOGS = {
    "desi": SPECTRA_CATALOG,
    "sdss": "hf://datasets/UniverseTBD/mmu_sdss_sdss",
    "provabgs": "hf://datasets/UniverseTBD/mmu_desi_provabgs",
    "hsc": "hf://datasets/UniverseTBD/mmu_hsc_pdr3_dud_22.5",
    "galaxies": "hf://datasets/Smith42/galaxies-with-hats",
    "galaxies_train": "hf://datasets/Smith42/galaxies-with-hats",
    "galaxies_validation": "hf://datasets/Smith42/galaxies-with-hats",
    "galaxies_test": "hf://datasets/Smith42/galaxies-with-hats",
}
SOURCE_SUBDIRS = {
    "galaxies": "train/train",
    "galaxies_train": "train/train",
    "galaxies_validation": "validation/validation",
    "galaxies_test": "test/test",
}
SOURCE_ID_COLUMNS = {
    "desi": "object_id",
    "sdss": "object_id",
    "provabgs": "object_id",
    "hsc": "object_id",
    "galaxies": "dr8_id",
    "galaxies_train": "dr8_id",
    "galaxies_validation": "dr8_id",
    "galaxies_test": "dr8_id",
}
CROSSMATCH_RADIUS_ARCSEC = 1.0
IMAGE_SHAPE = (3, 152, 152)
HSC_IMAGE_SHAPE = (5, 160, 160)
UNMATCHED_SOURCES = {"desi", "sdss", "hsc"}

MMU_ROOT = "mmu"
# Bumped whenever the record ORDER changes, since a saved stream position is
# an index into it: v2 bounded the unmatched-spectrum buffer (a fat cell's
# overflow now leads the cell instead of trailing it), v3 deals partitions to
# DP ranks instead of truncating and node-splitting them.
SOURCE_ASSEMBLY = "crossmatch_only_v3"
SOURCE_GRAPH_ASSEMBLY = "legacy_north_source_graph_v1"
# ~74.6 KiB per DESI spectrum row (measured, scripts/probe_stream_rss.py), so
# this is ~3.5k rows held per worker. Sized against the 68.7 GiB slurm cgroup
# at 8 workers x 2 ranks; raise it only with that arithmetic redone.
UNMATCHED_BUFFER_BYTES = 256 * 1024 * 1024

# Whole image cells are reserved, so image-bearing train/val objects are
# spatially disjoint. Stable global ownership assigns each unmatched spectrum
# partition to exactly one of those cells, preventing cross-split duplication.
VAL_PARTITIONS = 8
SPLIT_ORDER = 4
SPLIT_BUCKETS = 20
MATCH_INDEX_ENV = "ASTROPT3_MATCH_INDEX"

_IMAGE_SCALARS = ("ebv", "flux_g", "flux_r", "flux_z", "z_spec")
# Anchor columns already on the wire (0.01 MB against 645 MB of pixels) that
# carry loss-bearing scalar targets. fiberflux correlates only 0.64-0.67 with
# flux_*, so it is aperture concentration, not a rescale; psfdepth is an
# observing condition.
_ANCHOR_FREE_SCALARS = (
    "fiberflux_g",
    "fiberflux_r",
    "fiberflux_z",
    "psfdepth_g",
    "psfdepth_r",
    "psfdepth_z",
)
# The HSC image struct is band/flux/ivar/mask/psf_fwhm/scale. `ivar` alone is
# 47.0% of every HSC partition and is read nowhere, so project down to the two
# leaves attach_source consumes (`mask` measures 0.001% — free either way, but
# there is no reason to ask for it). HSC's own psf_fwhm stays off the wire:
# only the anchor's is a scalar target.
_HSC_IMAGE_LEAVES = ("flux", "band")
_HSC_IMAGE_COLUMNS = [f"image.{leaf}" for leaf in _HSC_IMAGE_LEAVES]
# i-band only. Every one of these is duplicated across grizy at pairwise
# correlations near 1 (extinction is exactly 1.0000), so one band is the field.
_HSC_SHAPE_MOMENTS = ("11", "22", "12")
_HSC_FREE_SCALARS = (
    *[f"{band}_cmodel_mag" for band in "grizy"],
    # magerr is fetched as a quality predicate and never becomes a target:
    # median 0.004-0.013 mag, i.e. it is the noise itself.
    *[f"{band}_cmodel_magerr" for band in "grizy"],
    "i_extendedness_value",
    *[f"i_sdssshape_shape{m}" for m in _HSC_SHAPE_MOMENTS],
    *[f"i_sdssshape_psf_shape{m}" for m in _HSC_SHAPE_MOMENTS],
)
# Project the spectrum struct's leaves, not the whole struct. `ivar` is 41% of
# every spectrum row's bytes and is not read by the streaming adapters;
# pyarrow accepts
# dotted leaf paths in read_row_group(columns=...) and returns a struct
# carrying only the children asked for. _spectrum_part whitelists the same
# three, so an unprojected read path cannot smuggle ivar back in.
_SPECTRUM_LEAVES = ("flux", "lambda", "mask")
_SPECTRUM_COLUMNS = [f"spectrum.{leaf}" for leaf in _SPECTRUM_LEAVES]
_SOURCE_COLUMNS = {
    "desi": [
        "object_id",
        "ra",
        "dec",
        "_healpix_29",
        *_SPECTRUM_COLUMNS,
        "Z",
        "ZERR",
        "ZWARN",
    ],
    "sdss": [
        "object_id",
        "ra",
        "dec",
        "_healpix_29",
        *_SPECTRUM_COLUMNS,
        "Z",
        "Z_ERR",
        "ZWARNING",
    ],
    "hsc": [
        "object_id",
        "ra",
        "dec",
        "_healpix_29",
        *_HSC_IMAGE_COLUMNS,
        *_HSC_FREE_SCALARS,
    ],
    "provabgs": [
        "object_id",
        "ra",
        "dec",
        "_healpix_29",
        "LOG_MSTAR",
        "Z_HP",
        "Z_MW",
        "TAGE_MW",
        "AVG_SFR",
        "TSNR2_BGS",
    ],
}
_ANCHOR_COLUMNS = [
    "object_id",
    "ra",
    "dec",
    "_healpix_29",
    "image",
    *_IMAGE_SCALARS,
    *_ANCHOR_FREE_SCALARS,
]
# Legacy's image struct is 99.97% `flux` — there is no second plane to drop,
# so it is not projected.


# -- decode: hub row -> record dict ------------------------------------------


def _stack_ragged(arr: np.ndarray) -> np.ndarray:
    """Recursively stack object arrays-of-arrays (arrow nested lists)."""
    if arr.dtype == object:
        return np.stack([_stack_ragged(np.asarray(x)) for x in arr])
    return arr


def _image_flux(value, expected_shape=IMAGE_SHAPE) -> np.ndarray:
    """Coerce nested lists / object arrays of band images to a dense cube."""
    arr = _stack_ragged(np.asarray(value)).astype(np.float32, copy=False)
    if arr.shape != expected_shape:
        raise ValueError(f"image flux has shape {arr.shape}, expected {expected_shape}")
    return arr


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"expected a numeric value, got {value!r}") from error


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"expected an integer value, got {value!r}") from error


def _finite(value) -> bool:
    return value is not None and math.isfinite(_as_float(value))


def _base(row) -> dict:
    return {
        "object_id": str(row["object_id"]),
        "ra": row["ra"],
        "dec": row["dec"],
        "_healpix_29": row["_healpix_29"],
    }


def _spectrum_part(row) -> dict:
    """The three leaves the sequencer reads, whatever else the row carries."""
    spectrum = row["spectrum"]
    missing = [leaf for leaf in _SPECTRUM_LEAVES if spectrum.get(leaf) is None]
    if missing:
        raise ValueError(f"spectrum row is missing {missing}")
    return {
        leaf: np.asarray(spectrum[leaf], dtype=bool if leaf == "mask" else np.float32)
        for leaf in _SPECTRUM_LEAVES
    }


def _attach_spectrum(record: dict, row) -> None:
    record["spectrum"] = _spectrum_part(row)
    for key in ("Z", "ZERR"):
        if _finite(row.get(key)):
            record[key] = _as_float(row[key])
    if row.get("ZWARN") is not None:
        record["ZWARN"] = bool(row["ZWARN"])


def _attach_free_scalars(record: dict, row, predicates: dict) -> None:
    """Promote already-fetched columns to scalar targets.

    A field failing its predicate is omitted, never defaulted. Grouping is
    all-or-nothing downstream: ``_scalar_value``
    returns ``None`` if any of a modality's ``record_keys`` is absent, so a
    dropped band drops its whole span rather than poisoning it.
    """
    for key, predicate in predicates.items():
        if _finite(row.get(key)) and predicate(_as_float(row[key])):
            record[key] = _as_float(row[key])


_ANCHOR_SCALAR_PREDICATES = {
    **{
        key: lambda value: True for key in ("fiberflux_g", "fiberflux_r", "fiberflux_z")
    },
    # ivar-like depth: log10(1+x) needs x >= 0, and 0 means no coverage
    **{
        key: lambda value: value > 0
        for key in ("psfdepth_g", "psfdepth_r", "psfdepth_z")
    },
}


def _attach_image(record: dict, row) -> None:
    bands = [str(b) for b in row["image"]["band"]]
    record["image"] = {
        "flux": _image_flux(row["image"]["flux"]),
        "band": bands,
    }
    for key in _IMAGE_SCALARS:
        if _finite(row.get(key)):
            record[key] = _as_float(row[key])
    _attach_free_scalars(record, row, _ANCHOR_SCALAR_PREDICATES)
    # Seeing is one value per band, keyed by band name rather than by position.
    # The record already carries the band list, and a positional
    # assumption would silently mis-key if a survey ever reorders its cube.
    fwhm = row["image"].get("psf_fwhm")
    if fwhm is not None and len(fwhm) == len(bands):
        for band, value in zip(bands, fwhm):
            if _finite(value) and 0 < _as_float(value) < 5:
                record[f"psf_fwhm_{band}"] = _as_float(value)


_HSC_MAX_MAGERR = 0.1


def _attach_hsc_free_scalars(record: dict, row) -> None:
    """Attach HSC scalars from the partner row already fetched.

    ``cmodel_magerr`` is fetched but NEVER stored as a target — its median is
    0.004-0.013 mag, so it is the measurement noise, not a property of the
    object. It earns its bytes as the magnitude's quality predicate instead.
    """
    for band in "grizy":
        mag, err = f"{band}_cmodel_mag", f"{band}_cmodel_magerr"
        if (
            _finite(row.get(mag))
            and 10 < _as_float(row[mag]) < 30
            and _finite(row.get(err))
            and _as_float(row[err]) < _HSC_MAX_MAGERR
        ):
            record[f"hsc_{mag}"] = _as_float(row[mag])
    extendedness = row.get("i_extendedness_value")
    if _finite(extendedness) and _as_float(extendedness) in (0.0, 1.0):
        record["hsc_extendedness"] = _as_float(extendedness)
    for moment in _HSC_SHAPE_MOMENTS:
        for prefix in ("sdssshape_shape", "sdssshape_psf_shape"):
            key = f"i_{prefix}{moment}"
            if _finite(row.get(key)):
                record[f"hsc_{key}"] = _as_float(row[key])


def attach_source(record: dict, source: str, row) -> None:
    """Attach one source-distinct partner row to an anchor record."""
    if source == "desi":
        _attach_spectrum(record, row)
        return
    if source == "sdss":
        spectrum = _spectrum_part(row)
        valid = spectrum["lambda"] > 0
        if not valid.any():
            raise ValueError("SDSS spectrum has no positive wavelengths")
        record["sdss_spectrum"] = {
            key: values[valid] for key, values in spectrum.items()
        }
        if (
            not bool(row.get("ZWARNING"))
            and _finite(row.get("Z"))
            and _as_float(row["Z"]) >= 0
        ):
            record["sdss_Z"] = _as_float(row["Z"])
        if _finite(row.get("Z_ERR")):
            record["sdss_Z_ERR"] = _as_float(row["Z_ERR"])
        if row.get("ZWARNING") is not None:
            record["sdss_ZWARNING"] = bool(row["ZWARNING"])
        return
    if source == "hsc":
        image = row.get("image")
        if image is not None:
            record["hsc_image"] = {
                "flux": _image_flux(image["flux"], HSC_IMAGE_SHAPE),
                "band": [str(b) for b in image["band"]],
            }
        _attach_hsc_free_scalars(record, row)
        return
    if source == "provabgs":
        if not (_finite(row.get("TSNR2_BGS")) and _as_float(row["TSNR2_BGS"]) > 0):
            return
        predicates = {
            "LOG_MSTAR": lambda value: 0 < value < 15,
            "Z_HP": lambda value: 0 <= value < 2,
            "Z_MW": lambda value: 0 <= value < 0.1,
            "TAGE_MW": lambda value: 0 <= value <= 14,
            "AVG_SFR": lambda value: 0 <= value < 1e4,
        }
        for key, predicate in predicates.items():
            if _finite(row.get(key)) and predicate(_as_float(row[key])):
                record[f"provabgs_{key}"] = _as_float(row[key])
        return
    if source == "galaxies" or source.startswith("galaxies_"):
        for key in GWH_FRACTION_FIELDS:
            if _finite(row.get(key)) and 0 <= _as_float(row[key]) <= 1:
                record[f"gwh_{key}"] = _as_float(row[key])
        return
    raise ValueError(f"unknown source adapter {source!r}")


def _source_id(source: str, row) -> str:
    value = row[SOURCE_ID_COLUMNS[source]]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif source == "sdss" and isinstance(value, str) and value.startswith("b'"):
        try:
            decoded = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            decoded = value
        if isinstance(decoded, bytes):
            value = decoded.decode("utf-8")
    return str(value).strip()


def source_only_record(source: str, row) -> dict:
    source_id = _source_id(source, row)
    record = {
        "object_id": f"{source}:{source_id}",
        "ra": row["ra"],
        "dec": row["dec"],
        "_healpix_29": row["_healpix_29"],
    }
    attach_source(record, source, row)
    return record


def decode_record(row) -> dict:
    """Convert one raw union-schema row into an ``ObjectSequencer`` record."""
    record = _base(row)
    image = row.get("image")
    spectrum = row.get("spectrum")
    has_image = image is not None and image.get("band") is not None
    has_spectrum = spectrum is not None and spectrum.get("flux") is not None
    if has_image:
        _attach_image(record, row)
    if has_spectrum:
        _attach_spectrum(record, row)
    if not (has_image or has_spectrum):
        raise ValueError(f"row {record['object_id']!r} has neither image nor spectrum")
    return record


# -- catalog partitions ------------------------------------------------------


def catalog_files(url: str) -> tuple[list[str], dict]:
    """Return HEALPix-ordered parquet paths and a cell-to-path mapping."""
    paths = cast(Any, importlib.import_module("hats.io.paths"))
    read_hats = cast(Any, importlib.import_module("hats.loaders.read_hats")).read_hats

    collection = read_hats(url)
    catalog = cast(Any, getattr(collection, "main_catalog", collection))
    files, by_cell = [], {}
    for pixel in catalog.get_healpix_pixels():
        rel = str(paths.pixel_catalog_file(catalog.catalog_base_dir, pixel)).replace(
            "hf://datasets/", "datasets/"
        )
        path = "hf://" + rel
        files.append(path)
        by_cell[(_as_int(pixel.order), _as_int(pixel.pixel))] = path
    return files, by_cell


def split_files(files: list, split: str, val_partitions: int = VAL_PARTITIONS) -> list:
    """Reserve the first K cells for validation and the remainder for train."""
    if not val_partitions:
        return list(files)
    reserved = min(val_partitions, max(1, len(files) // 5))
    return files[:reserved] if split == "val" else files[reserved:]


def split_of_cell(cell: tuple[int, int]) -> str:
    """Apply the common spatial split at a fixed nested HEALPix order."""
    order, pixel = cell
    if order < SPLIT_ORDER:
        raise ValueError(f"HEALPix order {order} is below split order {SPLIT_ORDER}")
    parent = pixel >> (2 * (order - SPLIT_ORDER))
    bucket = zlib.crc32(str(parent).encode()) % SPLIT_BUCKETS
    return "val" if bucket == 0 else "train"


def shuffled(files: list, seed: int, epoch: int) -> list:
    """Return a deterministic per-epoch partition permutation."""
    order = np.random.default_rng([seed, epoch]).permutation(len(files))
    return [files[i] for i in order]


def owned_by_rank(files: list, shard: int, num_shards: int) -> list:
    """Deal partitions round-robin to one DP rank.

    Replaces ``datasets.split_dataset_by_node``, which only assigns shards when
    ``num_shards % world_size == 0`` and otherwise silently degrades to "keep 1
    example out of world_size" — every rank then opens every partition and
    discards the rest, i.e. world_size times the wire bytes for the same
    tokens. Dealing the list ourselves cannot hit that path, and unlike
    truncating to a shard multiple it drops nothing (that cost 37 of 165 cells
    an epoch at dp=64). Ranks may differ by one partition.
    """
    return files[shard::num_shards] if num_shards > 1 else files


# -- match index -------------------------------------------------------------


def resolve_match_index(match_index: str | None = None) -> str | None:
    """Use the explicit index, then ``$ASTROPT3_MATCH_INDEX``."""
    return match_index or os.environ.get(MATCH_INDEX_ENV) or None


def is_source_graph(graph) -> bool:
    """Does this index need the multi-source stream rather than DESI-only?

    Schema 2 is the edge list and 3 the one-row-per-anchor pivot of the same
    graph; the layout does not change record order, only the source set does.
    Both the assembly tag and the stream dispatch ask this one question — they
    used to ask it separately, and adding schema 3 to only one of them sent a
    three-spoke index down the DESI-only path.
    """
    return graph.schema_version in (2, 3) and set(graph.partner_revisions) != {"desi"}


def assembly_and_revisions(match_index: str | None) -> tuple[str, dict]:
    """Resume-state tag plus the pinned source revisions behind it.

    Both come from one ``load_source_graph`` read. The graph is a 2M-row
    parquet, so loading it twice to answer two questions is not worth it.
    """
    from pathlib import Path

    resolved = resolve_match_index(match_index)
    if resolved is None or (
        not resolved.startswith("hf://") and not Path(resolved).exists()
    ):
        return SOURCE_ASSEMBLY, {}
    graph = load_source_graph(resolved)
    assembly = SOURCE_GRAPH_ASSEMBLY if is_source_graph(graph) else SOURCE_ASSEMBLY
    revisions = {"anchor": graph.anchor_revision, **dict(graph.partner_revisions)}
    return assembly, revisions


def source_assembly_for_index(match_index: str | None) -> str:
    """Return the resume-state tag implied by a resolved pointer index."""
    return assembly_and_revisions(match_index)[0]


def load_match_index(path: str):
    """Compatibility view of a DESI-only index for the v3 stream."""
    graph = load_source_graph(path)
    unsupported = set(graph.partner_revisions) - {"desi"}
    if unsupported:
        raise ValueError(
            f"crossmatch_only_v3 cannot stream sources {sorted(unsupported)}"
        )
    matches = {
        cell: {
            anchor_id: partners["desi"]
            for anchor_id, partners in anchors.items()
            if "desi" in partners
        }
        for cell, anchors in graph.matches.items()
    }
    spectra_of = {
        cell: sources.get("desi", set())
        for cell, sources in graph.partner_cells.items()
    }
    return matches, spectra_of


# -- crossmatch dataset ------------------------------------------------------


def _parquet_stream(files: list):
    """Open parquet through datasets only to derive its published features."""
    from datasets import load_dataset

    return load_dataset(
        "parquet", data_files=list(files), split="train", streaming=True
    )


def union_features(image_file: str, spectrum_file: str):
    """Derive the raw image ∪ spectrum schema from the published catalogs."""
    from datasets import Features

    image = cast(Any, _parquet_stream([image_file])).features
    spectrum = cast(Any, _parquet_stream([spectrum_file])).features
    if image is None or spectrum is None:
        raise ValueError("catalog parquet did not expose a feature schema")
    return Features({**cast(dict, image), **cast(dict, spectrum)})


def _rows(parquet_file, columns=None):
    """Yield rows without materializing a row group as Python objects."""
    for i in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(i, columns=columns)
        for j in range(table.num_rows):
            yield table.slice(j, 1).to_pylist()[0]


def _spectrum_only_rows(tables):
    """Yield each buffered spectrum row as an image-less record."""
    for table in tables:
        for i in range(table.num_rows):
            yield {**table.slice(i, 1).to_pylist()[0], "image": None}


def _crossmatch_examples(
    image_paths,
    match_json,
    spectra_paths,
    owned_spectra,
    matched_spectra_ids,
):
    """Yield pairs, unmatched images, and globally unmatched spectra once."""
    import fsspec
    import pyarrow as pa
    import pyarrow.parquet as pq

    pc = cast(Any, importlib.import_module("pyarrow.compute"))
    paired_globally = pa.array(sorted(map(str, matched_spectra_ids)), type=pa.string())
    for image_path, raw, spectrum_paths, owned in zip(
        image_paths, match_json, spectra_paths, owned_spectra
    ):
        try:
            wanted = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("invalid match-index partition JSON") from error

        needed = pa.array(sorted(map(str, wanted.values())), type=pa.string())
        owned = set(owned)
        matched_tables, buffered, buffered_bytes = [], [], 0
        for path in spectrum_paths:
            with fsspec.open(path, "rb") as file:
                parquet = pq.ParquetFile(file)
                for i in range(parquet.num_row_groups):
                    table = parquet.read_row_group(i)
                    ids = table["object_id"].cast(pa.string())
                    matched_tables.append(table.filter(pc.is_in(ids, needed)))
                    if path not in owned:
                        continue
                    rest = table.filter(pc.invert(pc.is_in(ids, paired_globally)))
                    # Matched spectra must stay resident (the image scan looks
                    # them up by id, in image order), but the unmatched ones
                    # only wait to be strided out. Buffering a whole cell's
                    # worth pinned up to 1.1 GiB per worker on the real
                    # catalogs -- 16 workers x that is what OOM-killed the
                    # crossmatch-only run at step 13,354. Past the budget,
                    # emit immediately instead of holding.
                    if buffered_bytes + rest.nbytes > UNMATCHED_BUFFER_BYTES:
                        yield from _spectrum_only_rows([rest])
                        continue
                    buffered.append(rest)
                    buffered_bytes += rest.nbytes

        matched = pa.concat_tables(matched_tables) if matched_tables else None
        spectra = (
            {}
            if matched is None
            else {
                str(matched["object_id"][i].as_py()): i for i in range(matched.num_rows)
            }
        )
        unmatched_count = sum(table.num_rows for table in buffered)
        pending = _spectrum_only_rows(buffered)

        with fsspec.open(image_path, "rb") as file:
            parquet = pq.ParquetFile(file)
            stride = (
                max(1, parquet.metadata.num_rows // unmatched_count)
                if unmatched_count
                else 0
            )
            for row_position, row in enumerate(_rows(parquet)):
                if stride and row_position % stride == 0:
                    spectrum_row = next(pending, None)
                    if spectrum_row is not None:
                        yield spectrum_row

                spectrum_id = wanted.get(str(row["object_id"]))
                if spectrum_id is None:
                    yield {
                        **row,
                        "spectrum": None,
                        "Z": None,
                        "ZERR": None,
                        "ZWARN": None,
                    }
                    continue

                index = spectra.get(str(spectrum_id))
                if index is None or matched is None:
                    continue  # match index and catalog revision disagree
                spectrum = matched.slice(index, 1).to_pylist()[0]
                yield {
                    **row,
                    "spectrum": spectrum["spectrum"],
                    "Z": spectrum.get("Z"),
                    "ZERR": spectrum.get("ZERR"),
                    "ZWARN": spectrum.get("ZWARN"),
                }

        yield from pending


def crossmatch_dataset(
    image_paths,
    match_json,
    spectra_paths,
    owned_spectra,
    matched_spectra_ids,
    features,
):
    """Build the sharded raw crossmatch dataset from resolved partition paths."""
    from datasets import IterableDataset

    return IterableDataset.from_generator(
        _crossmatch_examples,
        gen_kwargs={
            "image_paths": image_paths,
            "match_json": match_json,
            "spectra_paths": spectra_paths,
            "owned_spectra": owned_spectra,
            "matched_spectra_ids": matched_spectra_ids,
        },
        features=features,
    )


def _source_graph_examples(
    image_paths,
    match_json,
    source_partitions_json,
    owned_source_paths,
    matched_source_ids,
):
    """Yield one deterministic source-graph cell without materializing it."""
    import fsspec
    import pyarrow.parquet as pq

    globally_matched = {
        source: set(map(str, ids)) for source, ids in matched_source_ids.items()
    }
    for image_path, raw_matches, raw_partitions, owned in zip(
        image_paths, match_json, source_partitions_json, owned_source_paths
    ):
        try:
            wanted = json.loads(raw_matches)
            partitions = json.loads(raw_partitions)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("invalid source-graph partition JSON") from error

        needed = {
            source: {
                str(source_ids[source])
                for source_ids in wanted.values()
                if source in source_ids
            }
            for source in partitions
        }
        partner_rows: dict[str, dict[str, dict]] = {}
        for source in sorted(partitions):
            matched: dict[str, dict] = {}
            partner_rows[source] = matched
            owned_paths = set(owned.get(source, []))
            for spec in partitions[source]:
                path = spec["path"]
                with fsspec.open(path, "rb") as file:
                    parquet = pq.ParquetFile(file)
                    columns = (
                        ["dr8_id", "ra", "dec", "_healpix_29", *GWH_FRACTION_FIELDS]
                        if source == "galaxies" or source.startswith("galaxies_")
                        else _SOURCE_COLUMNS[source]
                    )
                    for row in _rows(parquet, columns):
                        source_id = _source_id(source, row)
                        if source_id in needed[source]:
                            if source_id in matched:
                                raise ValueError(
                                    f"duplicate {source} id {source_id!r} in fetched partitions"
                                )
                            matched[source_id] = row
                        elif (
                            source in UNMATCHED_SOURCES
                            and path in owned_paths
                            and source_id not in globally_matched[source]
                        ):
                            yield source_only_record(source, row)

        with fsspec.open(image_path, "rb") as file:
            parquet = pq.ParquetFile(file)
            for row in _rows(parquet, _ANCHOR_COLUMNS):
                anchor_id = str(row["object_id"])
                record = _base(row)
                _attach_image(record, row)
                for source, partner_id in sorted(wanted.get(anchor_id, {}).items()):
                    partner = partner_rows.get(source, {}).get(str(partner_id))
                    if partner is None:
                        warnings.warn(
                            f"match index points to missing {source} id "
                            f"{partner_id!r}; skipping source attachment",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        continue
                    attach_source(record, source, partner)
                yield record


def source_graph_dataset(
    image_paths,
    match_json,
    source_partitions_json,
    owned_source_paths,
    matched_source_ids,
):
    from datasets import IterableDataset

    return IterableDataset.from_generator(
        _source_graph_examples,
        gen_kwargs={
            "image_paths": image_paths,
            "match_json": match_json,
            "source_partitions_json": source_partitions_json,
            "owned_source_paths": owned_source_paths,
            "matched_source_ids": matched_source_ids,
        },
    )


def _source_catalog(source: str, revision: str) -> str:
    try:
        catalog = f"{SOURCE_CATALOGS[source]}@{revision}"
    except KeyError as error:
        raise ValueError(f"no catalog registered for source {source!r}") from error
    subdir = SOURCE_SUBDIRS.get(source)
    return f"{catalog}/{subdir}" if subdir else catalog


def _partition_owner(path: str, referencing: list, in_split: list):
    candidates = sorted(cell for cell in referencing if cell in in_split)
    if not candidates:
        return None
    return candidates[zlib.crc32(path.encode()) % len(candidates)]


def _source_graph_dataset(match_index, split, seed, epoch, shard, num_shards):
    graph = load_source_graph(match_index)
    image_catalog = f"{IMAGES_CATALOG}@{graph.anchor_revision}"
    _, image_by_cell = catalog_files(image_catalog)

    source_cells: dict[str, dict] = {}
    for source, revision in graph.partner_revisions.items():
        _, source_cells[source] = catalog_files(_source_catalog(source, revision))

    all_cells = sorted(graph.matches)
    missing = [cell for cell in all_cells if cell not in image_by_cell]
    if missing:
        raise ValueError(
            f"match index references {len(missing)} anchor partitions absent from "
            f"{image_catalog} (first: {missing[0]})"
        )
    in_split = [cell for cell in all_cells if split_of_cell(cell) == split]

    references: dict[tuple[str, str], list[tuple[int, int]]] = {}
    partition_specs: dict[tuple[int, int], dict[str, list[dict]]] = {}
    for cell in all_cells:
        per_source: dict[str, list[dict]] = {}
        for source, partner_cells in graph.partner_cells[cell].items():
            specs = []
            for partner_cell in sorted(partner_cells):
                try:
                    path = source_cells[source][partner_cell]
                except KeyError as error:
                    raise ValueError(
                        f"index references missing {source} partition {partner_cell}"
                    ) from error
                specs.append(
                    {"path": path, "order": partner_cell[0], "pixel": partner_cell[1]}
                )
                references.setdefault((source, path), []).append(cell)
            per_source[source] = specs
        partition_specs[cell] = per_source

    owners: dict[tuple[str, str], tuple[int, int]] = {}
    for key, cells_referencing in references.items():
        source, path = key
        if source not in UNMATCHED_SOURCES:
            continue
        partner_spec = next(
            spec
            for cell in cells_referencing
            for spec in partition_specs[cell][source]
            if spec["path"] == path
        )
        if split_of_cell((partner_spec["order"], partner_spec["pixel"])) != split:
            continue
        owner = _partition_owner(path, cells_referencing, in_split)
        if owner is not None:
            owners[key] = owner

    for (source, path), owner in owners.items():
        specs = partition_specs[owner].setdefault(source, [])
        if not any(spec["path"] == path for spec in specs):
            source_cell = next(
                cell for cell, value in source_cells[source].items() if value == path
            )
            specs.append(
                {"path": path, "order": source_cell[0], "pixel": source_cell[1]}
            )

    cells = owned_by_rank(shuffled(in_split, seed, epoch), shard, num_shards)
    matched_ids = {
        source: sorted(
            {
                str(source_ids[source])
                for cell_matches in graph.matches.values()
                for source_ids in cell_matches.values()
                if source in source_ids
            }
        )
        for source in graph.partner_revisions
    }
    return source_graph_dataset(
        image_paths=[image_by_cell[cell] for cell in cells],
        match_json=[json.dumps(graph.matches[cell]) for cell in cells],
        source_partitions_json=[json.dumps(partition_specs[cell]) for cell in cells],
        owned_source_paths=[
            {
                source: [
                    spec["path"]
                    for spec in specs
                    if owners.get((source, spec["path"])) == cell
                ]
                for source, specs in partition_specs[cell].items()
            }
            for cell in cells
        ],
        matched_source_ids=matched_ids,
    )


def _spectrum_owners(spectra_paths_by_cell: dict) -> dict:
    """Assign each spectrum partition to one stable image cell globally."""

    references: dict[str, list[tuple]] = {}
    for cell in sorted(spectra_paths_by_cell):
        for path in spectra_paths_by_cell[cell]:
            references.setdefault(path, []).append(cell)
    return {
        path: cells[zlib.crc32(path.encode()) % len(cells)]
        for path, cells in references.items()
    }


def _crossmatch_dataset(match_index, split, seed, epoch, shard, num_shards):
    graph = load_source_graph(match_index)
    unsupported = set(graph.partner_revisions) - {"desi"}
    if unsupported:
        raise ValueError(
            f"crossmatch_only_v3 cannot stream sources {sorted(unsupported)}"
        )
    image_catalog = IMAGES_CATALOG
    spectrum_catalog = SPECTRA_CATALOG
    # schema 1 carried no revisions; 2 and 3 both pin them
    if graph.schema_version in (2, 3):
        image_catalog += f"@{graph.anchor_revision}"
        spectrum_catalog += f"@{graph.partner_revisions['desi']}"
    image_files, image_by_cell = catalog_files(image_catalog)
    spectrum_files, spectrum_by_cell = catalog_files(spectrum_catalog)
    matches, spectra_of = load_match_index(match_index)

    all_cells = sorted(matches)
    missing = [cell for cell in all_cells if cell not in image_by_cell]
    if missing:
        raise ValueError(
            f"match index references {len(missing)} image partitions absent from "
            f"{image_catalog} (first: {missing[0]}); rebuild the index"
        )

    paths_by_cell = {
        cell: [spectrum_by_cell[s] for s in sorted(spectra_of[cell])]
        for cell in all_cells
    }
    owners = _spectrum_owners(paths_by_cell)
    in_split = split_files(all_cells, split)
    cells = owned_by_rank(shuffled(in_split, seed, epoch), shard, num_shards)
    return crossmatch_dataset(
        image_paths=[image_by_cell[cell] for cell in cells],
        match_json=[json.dumps(matches[cell]) for cell in cells],
        spectra_paths=[paths_by_cell[cell] for cell in cells],
        owned_spectra=[
            [path for path in paths_by_cell[cell] if owners[path] == cell]
            for cell in cells
        ],
        matched_spectra_ids={
            str(spectrum_id)
            for cell_matches in matches.values()
            for spectrum_id in cell_matches.values()
        },
        features=union_features(image_files[0], spectrum_files[0]),
    )


def open_stream(
    *,
    split: str = "train",
    seed: int = 0,
    epoch: int = 0,
    shard: int = 0,
    num_shards: int = 1,
    match_index: str | None = None,
):
    """Open one finite, deterministic epoch of the crossmatch-only corpus."""
    match_index = resolve_match_index(match_index)
    if match_index is None:
        raise ValueError(
            f"crossmatch-only MMU streaming requires match_index or ${MATCH_INDEX_ENV}"
        )

    # The DP split is the partition deal in owned_by_rank, not
    # split_dataset_by_node — see its docstring for why.
    graph = load_source_graph(match_index)
    if is_source_graph(graph):
        stream = _source_graph_dataset(
            match_index, split, seed, epoch, shard, num_shards
        )
        assembly = SOURCE_GRAPH_ASSEMBLY
    else:
        stream = _crossmatch_dataset(match_index, split, seed, epoch, shard, num_shards)
        assembly = SOURCE_ASSEMBLY
        stream = stream.map(decode_record)
    print(
        f"[data] open_stream {assembly} split={split} epoch={epoch} "
        f"shard={shard}/{num_shards} n_shards={stream.n_shards}",
        flush=True,
    )
    return stream
