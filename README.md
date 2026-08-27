# mmu-stream

Hub-and-spoke record streaming for HATS catalogs.

`mmu-stream` currently extracts the Legacy North five-spoke stream used by AstroPTv3. One stream has one hub and one or more independently indexed positional spokes; spokes never join each other and missing spokes do not exclude a hub row. Support for caller-selected hubs and adapters is experimental work that will be validated through AstroPTv3/MMU only.

## Install

Pin an immutable Git revision:

```bash
uv add "mmu-stream @ git+https://github.com/Smith42/mmu-stream@<commit-sha>"
```

Index construction needs the optional dependencies:

```bash
uv sync --extra index
uv run python -m mmu_stream.build_match_index --help
uv run python -m mmu_stream.merge_match_index --help
```

Incubation callers import from the defining module:

```python
from mmu_stream.streaming import open_stream
```

## Provenance

The initial implementation is extracted under AGPL-3.0 from `Smith42/astroPT3`, branch `mmu-corpus-expansion`, commit `b4559cbd4f72646febfbbc99fa7f149ff0c1f50d`:

- `astro/src/astropt3/data/streaming.py`
- `astro/src/astropt3/data/match_index.py`
- `astro/scripts/build_match_index.py`
- `astro/scripts/merge_match_index.py`

There is no compatibility guarantee before publication.
