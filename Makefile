# SPDX-FileCopyrightText: 2024-present Johann Rudi <jrudi@vt.edu>
#
# SPDX-License-Identifier: CC0-1.0

# Makefile for a uv-managed Python package
#
# Author: Johann Rudi <jrudi@vt.edu>
#
# Run 'make help' to list available targets. Adapt the settings to the project;
# the targets need no changes.

# Settings --------------------------------------------------------------------

# set the directories with Python files to format, lint, compile, and clean
PY_DIRS := src/dlk tests examples

# set the directory with the benchmarks, which 'make test' does not run
BENCH_DIR := benchmarks

# set the citation file to sync on version bumps; a missing file skips the sync
CITATION_FILE := CITATION.cff

# set the release notes of the current version
# (use '=' instead of ':=' to read the version only when needed)
RELEASE_NOTES = docs/releases/v$(VERSION).md

# set the regex of the release notes' title; 'release-body' prints what follows
# (write '#' as '\#')
RELEASE_NOTES_TITLE := ^\# Release v

# set the files and directories to remove with 'make clean'
CLEAN_FILES := site dist .pytest_cache

# set the files and directories to remove with 'make distclean', besides those
# of 'make clean'
DISTCLEAN_FILES := .venv

# set the tests to run; empty runs all
# (e.g., make test TESTS="tests/loss/test_x.py::test_y")
TESTS ?=

# Commands --------------------------------------------------------------------

# skip syncing so 'uv run' keeps the groups chosen at 'uv sync'
UV_RUN        := uv run --no-sync
ISORT         := $(UV_RUN) isort
BLACK         := $(UV_RUN) black
LINT          := $(UV_RUN) basedpyright
COMPILE       := $(UV_RUN) python -m compileall -q
PYTEST        := $(UV_RUN) pytest
LICENSE_CHECK := $(UV_RUN) reuse lint
DOCS          := $(UV_RUN) zensical
CFF           := uvx cffconvert --validate
BUILD         := uv build --no-sources
UV_VERSION    := uv version --no-sync
# bare version & no color codes (e.g., for a filename)
VERSION = $(shell NO_COLOR=1 $(UV_VERSION) --short)

SED := sed
RM  := rm -rf

.DEFAULT_GOAL := help
MAKEFLAGS += --no-print-directory

# Help target -----------------------------------------------------------------

.PHONY: help

help: ## Show this message
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-22s %s\n", $$1, $$2}'

# Format & static check targets -----------------------------------------------

.PHONY: format format-check lint license-check compile

format: ## Format Python files with isort and black
	$(ISORT) $(PY_DIRS)
	$(BLACK) $(PY_DIRS)

format-check: ## Check isort and black formatting without modifying files
	$(ISORT) --check $(PY_DIRS)
	$(BLACK) --check $(PY_DIRS)

lint: ## Type-check and lint with basedpyright
	$(LINT) $(PY_DIRS)

license-check: ## Check SPDX copyright and license information with reuse
	$(LICENSE_CHECK)

compile: ## Compile Python files to catch syntax errors
	$(COMPILE) $(PY_DIRS)

# Test targets ----------------------------------------------------------------

.PHONY: test testq testv testvv bench

test:   ## Run pytest (select with TESTS="<files or node ids>")
testq:  ## Run pytest quietly
testv:  ## Run pytest verbosely
testvv: ## Run pytest verbosely and show captured output

testq:  PYTEST_FLAGS := --quiet
testv:  PYTEST_FLAGS := --verbose
testvv: PYTEST_FLAGS := --verbose --capture=no

test testq testv testvv:
	$(PYTEST) $(PYTEST_FLAGS) $(TESTS)

bench: ## Run the benchmarks (e.g., PYTEST_FLAGS=--benchmark-autosave)
	$(PYTEST) $(PYTEST_FLAGS) $(BENCH_DIR)

# Documentation targets -------------------------------------------------------

.PHONY: docs serve

docs: ## Build the documentation site
	$(DOCS) build

serve: ## Serve the documentation site locally
	$(DOCS) serve

# Versioning targets ----------------------------------------------------------

.PHONY: version version-patch version-minor version-major

version: ## Print the current package version
	@echo $(VERSION)

version-patch: ## Bump the patch version and sync the citation file
version-minor: ## Bump the minor version and sync the citation file
version-major: ## Bump the major version and sync the citation file

version-patch version-minor version-major:
	$(UV_VERSION) --bump $(@:version-%=%)
	@test ! -f $(CITATION_FILE) || $(MAKE) citation

# Releasing targets -----------------------------------------------------------

.PHONY: citation citation-check build release-body

citation: ## Sync version and release date into the citation file
	@$(SED) -i.bak \
		-e 's|^version: .*|version: "$(VERSION)"|' \
		-e "s|^date-released: .*|date-released: \"$$(date -u +%F)\"|" \
		$(CITATION_FILE) && $(RM) $(CITATION_FILE).bak
	@grep -E '^(version|date-released):' $(CITATION_FILE)

citation-check: ## Validate the citation file against the CFF schema
	$(CFF) --infile $(CITATION_FILE)

build: ## Build the sdist and wheel into dist/
	$(BUILD)

# print the release notes of the current version without frontmatter and title,
# which is the body of the GitHub release; match the full title, since the SPDX
# header in the frontmatter also starts with "# "
release-body: ## Print the current version's release notes for the GitHub release
	@test -f $(RELEASE_NOTES) || { echo "missing release notes: $(RELEASE_NOTES)" >&2; exit 1; }
	@$(SED) '1,/$(RELEASE_NOTES_TITLE)/d' $(RELEASE_NOTES)

# Cleaning targets ------------------------------------------------------------

.PHONY: clean distclean

clean: ## Remove build output and Python caches
	$(RM) $(CLEAN_FILES)
	find $(PY_DIRS) -type d -name __pycache__ -prune -exec rm -rf {} +

distclean: clean ## Remove everything clean does, plus the virtual environment
	$(RM) $(DISTCLEAN_FILES)
