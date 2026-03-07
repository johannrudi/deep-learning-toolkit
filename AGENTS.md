# AGENTS.md

## Setup

- run `pip install -e .`
- run `pip install -e ".[dev]"`

## Tests

- run `make format` to format Python files in `dlk` with `black`
- run `make compile` to compile Python files in `dlk` and subdirectories
- run `make test` to run `pytest` across the codebase
- run `python -m dlk.nets.mlp`
- run `python -m dlk.nets.transformer1d`
- run `python -m dlk.nets.unet`
- run `python -m dlk.nets.conv1d`
- run `python -m dlk.nets.conv2d`
- run `python -m dlk.nets.efficientnet`
- treat module `if __name__ == "__main__"` blocks as test entry points

## Code rules

- add docstrings for classes and functions
- document args and returns using the Google style
- use `r"""..."""` docstrings for math-heavy notation when needed
- use descriptive parameter names (for example: `input_size`, `output_size`, `hidden_layers_sizes`)
- use `*_activation` and `*_kwargs` naming patterns for layer options
- validate input dimensions in `forward` methods with informative assertion messages
- start most Python comments (which begin with "#") with a lowercase verb
- if a comment starts with a noun, capitalize the first letter
- keep comments concise and without redundancies in Hemingway style
