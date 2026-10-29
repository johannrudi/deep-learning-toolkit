---
# SPDX-FileCopyrightText: 2024-present Johann Rudi <jrudi@vt.edu>
#
# SPDX-License-Identifier: Apache-2.0

Title: Keeping Files Licensed with reuse
Author: Johann Rudi
Co-Authored-By: Claude Opus 5.5
Date: 2026-10-28
tags:
  - licensing
  - reuse
  - dev
---

# Keeping Files Licensed with reuse

Every file states its copyright holder and license in an SPDX header, following the [REUSE specification]. `make license-check` runs `reuse lint`, locally and in CI, and fails until every file has one. For anything not covered here, see the [reuse tool documentation].

## Licenses

| Files | License |
|---|---|
| Everything not listed below: code, documentation, configuration | `Apache-2.0` |
| `Makefile`, shared with other repositories | `CC0-1.0` |
| Images in `docs/images/` | `CC-BY-4.0` |
| `examples/opt/diffusion_flow_matching_2d_checkerboard.ipynb`, adapted from [facebookresearch/flow_matching] | `CC-BY-NC-4.0` |

`LICENSE` holds the Apache-2.0 text for GitHub. `LICENSES/` holds one text per identifier used in a header.

## Add a file

```sh
uv run reuse annotate -c "Johann Rudi <jrudi@vt.edu>" -y 2024-present -l Apache-2.0 <files>
```

Keep the year `2024-present` on every file. Exceptions:

- **Notebooks, JSON files, and images** cannot hold a header that reuse reads; `REUSE.toml` covers them. For a new notebook, copy the first cell of an existing one. If `reuse annotate` writes a `.license` file, delete it and add an entry to `REUSE.toml` instead.
- **Generated files**: the release-notes scripts write the header into `CHANGELOG.md`, `docs/releases/index.md`, and new release notes. Do not annotate these.
- **Unknown file types**, such as `.cff`: pass `--style python` for `#` comments.

## Check

```sh
make license-check
uv run reuse lint --lines  # one problem per line
```

## Contributors

Add a contributor's `SPDX-FileCopyrightText` line only for a substantial contribution, such as a new file or function. Small fixes create no copyright. Running `reuse annotate` with their `-c` adds the line.

## Change a license or copyright holder

`reuse annotate` adds tag lines and never removes them, so edit the existing line by hand. A new identifier needs its text in `LICENSES/`:

```sh
uv run reuse download <identifier>
```

Remove a text from `LICENSES/` once no file uses it. A new holder also goes into `REUSE.toml` and `SPDX_COPYRIGHT` in `.agents/skills/write-release-notes/scripts/`.

<!-- REFERENCES -->

[REUSE specification]: https://reuse.software/spec/
[reuse tool documentation]: https://reuse.readthedocs.io/
[facebookresearch/flow_matching]: https://github.com/facebookresearch/flow_matching
