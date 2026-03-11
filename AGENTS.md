# AGENTS.md

## Setup

- use Python `>=3.11`
- run `pip install -e .`
- run `pip install -e ".[dev]"`
- run `pip install -e ".[test]"` when you only need test dependencies
- run `pip install -e ".[kde]"` when you work on KDE features

## Tests

- run `make format` to format Python files in `dlk` and `tests` with `black` and `isort`
- run `make compile` to compile Python files in `dlk` and `tests`
- run `make lint` to run `basedpyright` across `dlk` and `tests`
- run `make test` to run `pytest` across the codebase
- run `make testq` to run `pytest -q` across the codebase
- run `make testv` to run `pytest -v` across the codebase
- run `make testvv` to run `pytest -sv` across the codebase

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
