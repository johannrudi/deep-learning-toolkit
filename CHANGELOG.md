<!-- Generated from docs/releases by .agents/skills/write-release-notes/scripts/collect_changelog.sh -->
<!-- Do not edit by hand; rerun that script after adding a release. -->

# Changelog

Release notes for the *Deep Learning Toolkit*, newest first. Each entry links to
its full notes, which add the per-commit changelog and the changed-file
histograms.

## [v0.5.0](https://github.com/johannrudi/deep-learning-toolkit/releases/tag/v0.5.0) · 2026-08-07

This version introduces full Distributed Data Parallel (DDP) support for training on multi-GPU systems. `dlk.opt` now provides building blocks for distributed training, rank-aware train loops for both supervised learning and GAN training, and distributed profiling to measure performance scaling. The release also includes distributed checkpoint loading and a comprehensive guide to distributed training workflows.

Alongside DDP, this release fixes a CPU floating-point nondeterminism issue that caused regression tests to fail randomly across different hardware, and improves type hints for learning rate schedulers. Build improvements include support for PyTorch version selection via dependency groups (CPU-only, CUDA 12.6, 12.8, 13.0) and a streamlined release process with automated changelog generation.

[Full notes](docs/releases/v0.5.0.md) ·
[Compare](https://github.com/johannrudi/deep-learning-toolkit/compare/v0.4.1...v0.5.0)

## [v0.4.1](https://github.com/johannrudi/deep-learning-toolkit/releases/tag/v0.4.1) · 2026-06-01

This version separates evaluation from training. `dlk.eval` is a new module for prediction and evaluation from trained models, and it starts with the diffusion case, which previously had no home outside the training loop.

The type hint for validation callbacks passed into the training loops accepts more shapes now, so applications whose validation function returns something other than the previously declared type no longer need a cast to satisfy the linter.

[Full notes](docs/releases/v0.4.1.md) ·
[Compare](https://github.com/johannrudi/deep-learning-toolkit/compare/v0.4.0...v0.4.1)

## [v0.4.0](https://github.com/johannrudi/deep-learning-toolkit/releases/tag/v0.4.0) · 2026-05-27

This version adds diffusion models to the toolkit. `dlk.opt.train_diffusion` trains them on top of the `flow-matching` package, `dlk.nets.diffusion` holds the network side, and an example notebook works through a 2D checkerboard target end to end.

Two smaller additions change how an application is assembled. `dlk.plt` is a new module for plotting, starting with a loss plot, and `dlk.opt` can now build optimizers and learning-rate schedulers from a parameters dictionary instead of constructing them by hand. GAN training keeps its interface while its loss functions were simplified underneath.

One default moved: `dlk.mgmt` writes and reads parameter files as YAML rather than TOML. Applications that pass a file name keep working; applications that relied on the extension chosen for them need to follow. Progress bars now switch themselves off in noninteractive runs, which keeps log files readable when training under a batch scheduler.

[Full notes](docs/releases/v0.4.0.md) ·
[Compare](https://github.com/johannrudi/deep-learning-toolkit/compare/v0.3.1...v0.4.0)

## [v0.3.1](https://github.com/johannrudi/deep-learning-toolkit/releases/tag/v0.3.1) · 2026-03-19

This version changes nothing an application can observe. It follows v0.3.0 by a day and prepares the repository for being public: the README was corrected and rewritten in places, and the CI workflows moved to Node.js 24, ahead of GitHub Actions deprecating Node 20.

[Full notes](docs/releases/v0.3.1.md) ·
[Compare](https://github.com/johannrudi/deep-learning-toolkit/compare/v0.3.0...v0.3.1)

## [v0.3.0](https://github.com/johannrudi/deep-learning-toolkit/releases/tag/v0.3.0) · 2026-03-18

This version is the first published to PyPI, and it is the release in which the toolkit acquired its quality gate. Continuous integration now runs formatting, compilation, tests, and the `basedpyright` linter on every push, and the codebase was reviewed module by module to satisfy it: `nets`, `opt`, `loss`, and `metrics` all gained docstrings and type declarations, and the type definitions that networks share moved into `dlk/nets/utils.py`.

Two changes are visible in imports. The configuration module `config` became `dlk.mgmt`, which now loads and saves parameter files, adds its arguments to an `argparse` parser, and owns logging setup. `dlk.mode` is new and carries the execution modes an application selects between, such as train, predict, and eval.

`dlk.metrics` gained the Hellinger distance in several forms: histogram-based, kernel-density-based through the optional `torchkde` package, and a scale-invariant variant. The package no longer depends on numpy, since torch does that work, and the minimum Python version rose to 3.11 so that `tomllib` is available from the standard library.

[Full notes](docs/releases/v0.3.0.md) ·
[Compare](https://github.com/johannrudi/deep-learning-toolkit/compare/v0.2.0...v0.3.0)

## [v0.2.0](https://github.com/johannrudi/deep-learning-toolkit/releases/tag/v0.2.0) · 2026-03-06

This version is the first tagged release of the toolkit. It gathers the networks, losses, training loops, and metrics that had accumulated in the repository into the `dlk` package, whose name dates from shortly before this tag, when the directory `dlkit` was renamed.

An application importing `dlk` at this point gets training loops for feed-forward networks and GANs, with checkpointing, learning-rate schedulers, transform hooks for inputs and targets, and a profiler that wraps the same loops. The network library covers MLPs and MLPResNets with optional attention, 1D and 2D convolutional networks with residual connections, UNets in both dimensions, encoder and decoder nets derived from them, autoencoders, EfficientNet, and transformers for time-series and channel-wise inputs. GAN training is served by Wasserstein, hinge, and least-squares losses, with several gradient-penalty variants for Wasserstein critics. `dlk.metrics` computes Wasserstein distance, maximum mean discrepancy, and Sinkhorn divergence.

The changelog in these notes folds 123 commits into one bullet per capability, since no earlier release exists to diff against.

[Full notes](docs/releases/v0.2.0.md)
