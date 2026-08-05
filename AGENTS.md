# AGENTS.md

## Setup

- use Python `>=3.11`
- use `uv` version `>=0.12.1,<0.13`
- run `uv sync` to create `.venv` and install the default `dev` dependency group
- run `uv sync --all-extras` to also install the `diffusion` and `kde` extras
- run `uv lock` and commit `uv.lock` after changing dependencies in `pyproject.toml`
- do not run `pip install -e ".[dev]"`; `dev` is a dependency group, not an extra

## Tests

- run `make format` to format Python files in `dlk` and `tests` with `black` and `isort`
- run `make format-check` to check `isort` and `black` formatting in `dlk` and `tests` without modifying files
- run `make compile` to compile Python files in `dlk` and `tests`
- run `make lint` to run `basedpyright` across `dlk` and `tests`
- run `make test` to run `pytest` across the codebase
- run `make testq` to run `pytest -q` across the codebase
- run `make testv` to run `pytest -v` across the codebase
- run `make testvv` to run `pytest -sv` across the codebase

## CI

- `.github/workflows/ci.yml` runs `make compile`, `make format-check`, `make lint`, and `make test` on Python 3.11
- CI sets `UV_LOCKED=1`; a stale `uv.lock` fails the build

## Release

- run `make version` to read the current version
- run `make version-patch` (or `version-minor`, `version-major`) to bump `project.version` in `pyproject.toml` and sync `CITATION.cff`
- `uv version` also updates `uv.lock`; commit `pyproject.toml`, `uv.lock`, and `CITATION.cff` together
- do not add a version to `dlk/__init__.py`; `__version__` is read from the installed package metadata
- run `uvx cffconvert --validate` after editing `CITATION.cff`
- run `uv build --no-sources` to build the sdist and wheel
- tag the release commit as `v{version}` and push the tag
- publish by creating a GitHub release; CI uploads to PyPI with trusted publishing

## Git

- use git worktrees only with the user's permission; the user tracks changes normally with git branches

## Code rules

- use `r"""..."""` docstrings for math notation when needed
- use descriptive parameter names (for example: `input_size`, `output_size`, `hidden_layers_sizes`)
- use `*_activation` and `*_kwargs` naming patterns for layer options
- validate input dimensions in `forward` methods with informative assertion messages

## Documentation

- add docstrings for classes and functions
- document arguments, returns, etc. using the Google Python Style Guide

## Comments in code

- overall, keep comments concise and without redundancies in Hemingway style
- start most Python comments (which begin with "#") with a lowercase verb
- but if a comment starts with a noun, capitalize the first letter
