# MMU Hub-and-Spoke Streaming Extraction Plan

## Goal

Extract AstroPTv3’s MMU/HATS hub-and-spoke record-streaming implementation into the standalone `mmu-stream` repository without changing behavior, prove parity through AstroPTv3, then generalize the package to one HATS hub with one or more independent positional spokes.

## Reference implementation

- Source repository: `https://github.com/Smith42/astroPT3`
- Source branch: `mmu-corpus-expansion`
- Source commit: `b4559cbd4f72646febfbbc99fa7f149ff0c1f50d`
- Target repository: `https://github.com/Smith42/mmu-stream`
- Architecture reference: `astro/docs/adr/0013-legacy-centred-mmu-expansion.md`

The reference topology is one rooted star per stream:

- one HATS hub/anchor;
- one or more independently indexed positional spokes;
- no spoke-to-spoke indexes;
- no complete N-way match requirement;
- one anchored graph per `open_stream()` call;
- the supplied graph, not sampling weights, defines the corpus.

## Non-goals

- No streaming-engine rewrite or optimization during extraction.
- No arbitrary join graph, spoke-to-spoke join, or multi-hub mixing engine.
- No source weights or standalone spoke scans.
- No AstroPT packing, batching, retry, checkpoint policy, transforms, or model code in the package.
- No package-root re-exports.
- No registry publication, release tags, or compatibility promise.
- No broad cleanup, renaming, or dead-code removal.

---

## Phase 0 — Establish the source-graph baseline

1. Assign named owners for:
   - implementation;
   - live/performance validation;
   - cross-repository pin and cutover coordination.
2. Create coordinated extraction branches in both repositories.
3. Record the exact source commit above and these original paths:
   - `astro/src/astropt3/data/streaming.py`;
   - `astro/src/astropt3/data/match_index.py`;
   - `astro/scripts/build_match_index.py`;
   - `astro/scripts/merge_match_index.py`.
4. Pin the current Legacy North five-spoke index and all catalog revisions used by the reference branch.
5. Capture the pre-extraction baseline:
   - AstroPT CPU suite;
   - `count_params.py`;
   - 50-step CPU smoke test;
   - live five-spoke decode and exact resume;
   - schema-v2 edge indexes and equivalent schema-v3 merged index;
   - current training-machine throughput using the existing five-spoke configuration, worker count, and DP layout.
6. Save deterministic stream evidence:
   - source-graph assembly tag and revisions;
   - first fixed record window and stream state;
   - source composition and fetched-only records;
   - train/validation, rank, and worker ownership;
   - stream `n_shards`.
7. Record benchmark environment details:
   - machine/node and network link;
   - worker count and DP layout;
   - workload/config commit and exact index;
   - warm-up interval;
   - repeated-run timing/throughput distribution;
   - stalls, retries, memory, and stream `n_shards`.

ADR 0013 remains Proposed; extraction parity does not close its remaining GPU, resume/no-replay, or learning-evidence gates.

**Exit:** Baseline evidence is saved and all three owner roles are assigned.

---

## Phase 1 — Scaffold `mmu-stream`

Create:

```text
pyproject.toml
src/mmu_stream/__init__.py
src/mmu_stream/streaming.py
src/mmu_stream/match_index.py
src/mmu_stream/build_match_index.py
src/mmu_stream/merge_match_index.py
tests/test_streaming.py
tests/test_match_index.py
```

Configure:

- Python `>=3.11`;
- hatchling build backend;
- package name `mmu-stream`;
- placeholder metadata version `0.0.0`;
- core dependencies:
  - `numpy>=1.26`
  - `datasets>=4.3`
  - `hats`
  - `pyarrow`
  - `fsspec`
  - `huggingface-hub`
- optional `index` extra containing direct dependencies `lsdb`, `dask`, and `pandas`;
- development dependency `pytest>=8.0`;
- `network` pytest marker;
- checked-in `uv.lock`.

Keep `src/mmu_stream/__init__.py` empty. Incubation callers import from the defining modules.

Update `README.md` with:

- the one-hub/one-or-more-spokes topology;
- current AstroPT Legacy North five-spoke validation scope;
- experimental arbitrary-hub intent within that fixed topology;
- installation from a Git revision;
- index-extra and both module-CLI usages;
- source branch, commit, and path provenance;
- no compatibility guarantee before publication.

**Exit:** A wheel builds; the stream and match-index modules import in a clean core environment, and the index CLIs import in the `index` environment.

---

## Phase 2 — Copy the source-graph implementation

1. Copy `astro/src/astropt3/data/match_index.py` to `src/mmu_stream/match_index.py`.
2. Preserve `MatchGraph` and schema-v1, schema-v2, and schema-v3 loading and validation.
3. Copy `astro/src/astropt3/data/streaming.py` to `src/mmu_stream/streaming.py` and change only package import paths plus the minimum needed to remove the AstroPT runtime dependency.
4. Copy only `GWH_FRACTION_FIELDS` into the extracted module rather than importing AstroPT’s torch-based scalar registry. Do not copy scalar transforms or model code.
5. Preserve source-graph behavior and constants, including:
   - one hub with one or more spokes;
   - deterministic split, shuffle, rank ownership, and worker sharding;
   - matched attachment and absent-spoke handling;
   - fetched-only unmatched ownership;
   - exact `datasets` resume state;
   - assembly/revision fingerprinting;
   - legacy DESI-only index compatibility.
6. Copy `astro/scripts/build_match_index.py` to `src/mmu_stream/build_match_index.py`.
7. Preserve one-hub/one-spoke-per-run LSDB positional crossmatching, pinned provenance, partition alignment, and schema-v2 output. Keep LSDB dynamically imported off the core streaming path.
8. Copy `astro/scripts/merge_match_index.py` to `src/mmu_stream/merge_match_index.py`.
9. Preserve schema-v2 edge-list to schema-v3 one-row-per-hub pivoting, null blocks for absent spokes, and duplicate-spoke rejection.
10. Record AGPL-3.0 relicensing authorization and exact AstroPT source provenance in the extraction commit.

Do not optimize, rename, generalize, or remove the legacy dispatch path in this phase.

**Exit:** The copied stream, match-index module, and both index CLIs import without AstroPTv3.

---

## Phase 3 — Establish package-owned tests

1. Copy only index validation and deterministic ownership tests from the reference branch.
2. Update imports to `mmu_stream`.
3. Do not create or maintain a synthetic/fake package stream. End-to-end row assembly and resume checks use the real catalog/index path and remain marked `network`.
4. Keep these offline checks in the package:
   - schema-v2 builder schema, ID normalization, aligned cells, and empty/full dtype parity;
   - schema-v2 spoke-directory loading;
   - schema-v2/schema-v3 `MatchGraph` equivalence;
   - mixed revision, invalid separation/radius, non-positional join, and duplicate-spoke rejection;
   - common spatial split and deterministic partition ownership;
   - rank ownership without dropped cells;
   - legacy index compatibility and match-index resolution.
5. Keep survey decoding, attachment, packing, resume, and model integration checks in AstroPT.
6. Keep live catalog/index tests marked `network`; do not run them in default offline CI.

Checks:

```bash
uv sync --extra index --group dev
uv run pytest -m "not network"
uv run python -m mmu_stream.build_match_index --help
uv run python -m mmu_stream.merge_match_index --help
uv run --extra index python -c "import lsdb, dask, pandas"
```

**Exit:** Package unit tests pass without AstroPT installed, and the package contains no synthetic stream fixture.

---

## Phase 4 — Publish an immutable incubation commit

1. Commit and push the complete graph implementation to `Smith42/mmu-stream`.
2. Capture the exact commit SHA.
3. Verify installation from that SHA in clean core and index environments.
4. Do not create a release or tag.

**Exit:** The package SHA is remotely available and reproducibly installable.

---

## Phase 5 — Migrate AstroPTv3 through one adapter module

1. Add `mmu-stream` as an AstroPT dependency pinned to the exact Git SHA.
2. Update `astro/uv.lock`.
3. Allow local development with an editable sibling installation without changing the committed Git pin.
4. Retain `astropt3.data.streaming` as AstroPT’s adapter module. Initially make it a thin wrapper over the package parity interface so existing callers keep one local seam.
5. Preserve the assembly/revision fingerprint exactly; a package move alone must not invalidate saved stream state.
6. Update Astro tests and monkeypatch targets to cross the adapter seam or the package module deliberately.
7. Keep AstroPT application-level stream tests behind the adapter seam.
8. Move the record-to-sequence live assertion into a small Astro integration test.
9. Delete Astro’s duplicate engine/index implementation after cutover:
   - `src/astropt3/data/match_index.py`;
   - package-owned implementation inside `src/astropt3/data/streaming.py`;
   - `scripts/build_match_index.py`;
   - `scripts/merge_match_index.py`;
   - package-owned portions of stream/index tests.
10. Update `scripts/build_remaining_spokes.sh`, `scripts/probe_stream_rss.py`, telemetry imports, and index command documentation to use the package modules.
11. Update references in `AGENTS.md`, ADRs 0006/0013/0014, architecture/training docs, and comments.
12. Remove Astro dependencies only when no direct Astro import remains. Retain dependencies needed by the Astro adapter and its tests.

The thin adapter is deliberate: Phase 8 moves survey-specific interpretation behind it without changing every loader, eval path, and test again.

**Exit:** AstroPT has one local streaming seam and no duplicate graph-streaming or index implementation.

---

## Phase 6 — Run the parity gate

### Package

- Offline package suite passes.
- Optional index environment imports.
- Schema-v2 and schema-v3 indexes load to the same `MatchGraph`.
- Every graph has one pinned hub and one or more positional spokes.
- Each spoke is independently buildable; no spoke-to-spoke edge exists.
- Hub rows attach every available spoke without requiring complete matches.
- Eligible fetched-only rows are emitted exactly once and remain split/rank/worker disjoint.
- Ineligible spokes do not emit unmatched rows.
- The live stream preserves record order, values, `n_shards`, state, and exact continuation.
- Source assembly and revision fingerprints remain unchanged.

### AstroPT

Run from `astro/`:

```bash
uv run pytest
uv run python scripts/count_params.py
uv run python -m astropt3.train_smoke \
  --config configs/model/test-tiny.yaml \
  --steps 50 \
  --assert-decrease
```

Also verify:

- nanotron loader shape/state tests;
- retry/rebuild tests;
- eval/generation stream-adapter tests;
- source-distinct modality and family-loss tests;
- live five-spoke record-to-sequence integration;
- package Git pin is the intended immutable SHA;
- a fixed before/after record window and resume state are identical.

### Performance

On the current training environment:

1. Use the same five-spoke config, worker count, catalog revisions, merged index, DP layout, and measurement window as the baseline.
2. Ignore initial prefetch-buffer drain.
3. Repeat the fixed workload enough to estimate normal variance.
4. Compare sustained step time, tokens/second, link utilization, memory, retries, stalls, source composition, and `n_shards`.
5. Fail on a meaningful regression outside baseline noise or any new stall/error mode.

**Exit:** All parity gates pass. Otherwise, keep the Astro cutover branch unmerged.

---

## Phase 7 — Atomic cutover

1. Ensure the referenced `mmu-stream` SHA is present on the remote.
2. Merge the package work first.
3. Re-run AstroPT validation against that immutable SHA.
4. Merge the AstroPT dependency/adapter switch and duplicate-engine deletion together.
5. Do not ship a feature flag, fallback import, or dormant duplicate.
6. Handle post-merge failures by fixing forward; no formal rollback artifact is prepared.

**Exit:** AstroPTv3 runs solely through the package graph engine.

---

## Phase 8 — Generalize the adapter seam after parity

Run a design-it-twice exercise comparing at least:

1. callback-based hub/spoke row adaptation;
2. one source-graph adapter object;
3. a hub adapter plus a source-keyed set of spoke adapters.

Compare each by:

- interface depth;
- locality;
- adapter burden;
- deterministic/resume semantics;
- support for absent spokes and fetched-only eligibility;
- leakage of Arrow, HATS, or Astro-specific details.

The package module should keep graph loading, partition planning, ownership, fetching, assembly, and resumability behind a small interface. Place the seam at source interpretation.

The generalized scope is limited to:

- exactly one HATS hub per stream;
- one or more independently indexed positional spokes;
- a graph-selected spoke set with no source weights;
- caller-provided catalog locations, ID extraction, column projection, hub decoding, and spoke attachment;
- caller-provided fetched-only eligibility per spoke, with package-owned exactly-once ownership mechanics;
- deterministic split, shuffle, rank/worker sharding, and resumable `datasets` state;
- independent spoke-index creation, graph consumption, and optional schema-v3 merge.

Move these Astro-specific concerns into `astropt3.data.streaming` adapters:

- catalog URLs, subdirectories, and `ASTROPT3_MATCH_INDEX`;
- source ID columns and canonicalization;
- projected columns;
- Legacy hub decoding;
- DESI, SDSS, HSC, PROVABGS, and galaxies-with-hats attachment rules;
- image/spectrum shapes and quality predicates;
- `GWH_FRACTION_FIELDS` and scalar rules;
- fetched-only eligibility for DESI, SDSS, and HSC;
- Astro assembly tags and telemetry names.

Do not add multi-hub mixing, spoke-to-spoke joins, arbitrary join graphs, plugin registration, alternate transports, source weights, standalone spoke scans, or Astro batching/retry behavior.

Validate the generalized package through the live stream and the full AstroPT regression/parity gates. Documentation must state that arbitrary-hub support is experimental and validated only through AstroPT’s MMU source graph.

Update AstroPT’s exact Git pin whenever generalized changes break the previous commit.

**Exit:** The package interface expresses one hub with one-or-more spokes, AstroPT owns survey interpretation, and all parity gates still pass.

---

## Definition of done

- `mmu-stream` is independently installable from an immutable Git SHA.
- It has no AstroPT runtime dependency.
- It owns the graph-streaming engine, `MatchGraph`/index loading, spoke-index builder, index merger, and package-level tests.
- Its topology is one hub with one or more independent positional spokes.
- AstroPT owns survey adapters, batching, retry/resume policy around packed batches, packing, and integration tests.
- AstroPT has no duplicate production graph-streaming or index implementation.
- Automated, live, resume, and performance parity gates pass on the five-spoke reference graph.
- Provenance and AGPL-3.0 relicensing are recorded.
- Arbitrary-hub support is documented as experimental and AstroPT/MMU-validated only.
- Publication, release versions, and compatibility policy remain deferred.
