# mmu-stream

Streams one HATS hub with one or more independently indexed positional
spokes. Spokes never join each other, and a hub row does not require every
spoke to match.

The current implementation reproduces AstroPTv3's Legacy North five-spoke
stream. Broader catalog support remains experimental.

## Install

Pin an immutable Git revision:

```bash
uv add "mmu-stream @ git+https://github.com/Smith42/mmu-stream@<commit-sha>"
```

## Stream

```python
from mmu_stream.streaming import open_stream

stream = open_stream(match_index="/path/to/match-index")
```

`open_stream` also reads the index path from `ASTROPT3_MATCH_INDEX`.

## Build indexes

Install the offline index dependencies, build each hub-to-spoke index
separately, then optionally merge them into one row per hub object:

```bash
uv sync --extra index
uv run python -m mmu_stream.build_match_index --help
uv run python -m mmu_stream.merge_match_index --help
```

## Development

```bash
uv sync --extra index --group dev
uv run pytest -m "not network"
```

Live tests require `ASTROPT3_MATCH_INDEX` and access to the referenced catalogs.

## Provenance

Extracted under AGPL-3.0 from `Smith42/astroPT3` branch
`mmu-corpus-expansion` at
`b4559cbd4f72646febfbbc99fa7f149ff0c1f50d`.

There is no compatibility guarantee before publication.
