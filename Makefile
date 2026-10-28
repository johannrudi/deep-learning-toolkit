# Makefile for this package
#
# Author: Johann Rudi <jrudi@vt.edu>
#
# Run 'make help' to list available targets.

# Shell programs
RM          := rm -rf
RSYNC       := rsync -aiz
RSYNC_LOCAL := rsync -ai
SED         := sed
TEST        := test

# Python commands
PY_FORMAT        := uv run --no-sync black
PY_FORMAT_IMPORT := uv run --no-sync isort
PY_LINT          := uv run --no-sync basedpyright
PY_COMPILE       := uv run --no-sync python -m compileall -q -f
PY_TEST          := uv run --no-sync pytest
PY_LICENSE_CHECK := uv run --no-sync reuse lint
PY_VERSION       := uv version --no-sync
# bare version & no color codes (e.g., for a filename):
PY_VERSION_READ  := NO_COLOR=1 $(PY_VERSION) --short

# Documentation generator commands
DOCS_BUILD := uv run --no-sync zensical build
DOCS_SERVE := uv run --no-sync zensical serve

# set directories
PACKAGE_DIR  := dlk
TESTS_DIR    := tests
EXAMPLES_DIR := examples
RELEASES_DIR := docs/releases
SITE_DIR     := site
SITE_DEST    := _TODO_

# set the tests to run; empty runs all (e.g., make test TESTS="tests/loss/test_x.py::test_y")
TESTS ?=

# set files
CITATION_FILE := CITATION.cff

# Help target -----------------------------------------------------------------

.PHONY: help

help: ## Show this message
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-22s %s\n", $$1, $$2}'

# Format & static check targets -----------------------------------------------

.PHONY: format format-check lint license-check compile

format-check: ## Check isort and black formatting without modifying files
	$(PY_FORMAT_IMPORT) --check $(PACKAGE_DIR)
	$(PY_FORMAT) --check $(PACKAGE_DIR)
	@echo
	$(PY_FORMAT_IMPORT) --check $(TESTS_DIR)
	$(PY_FORMAT) --check $(TESTS_DIR)
	@echo
	$(PY_FORMAT_IMPORT) --check $(EXAMPLES_DIR)
	$(PY_FORMAT) --check $(EXAMPLES_DIR)

format: ## Format Python files with isort and black
	$(PY_FORMAT_IMPORT) $(PACKAGE_DIR)
	$(PY_FORMAT) $(PACKAGE_DIR)
	@echo
	$(PY_FORMAT_IMPORT) $(TESTS_DIR)
	$(PY_FORMAT) $(TESTS_DIR)
	@echo
	$(PY_FORMAT_IMPORT) $(EXAMPLES_DIR)
	$(PY_FORMAT) $(EXAMPLES_DIR)

lint: ## Type-check and lint with basedpyright
	$(PY_LINT) $(PACKAGE_DIR)
	$(PY_LINT) $(TESTS_DIR)
	$(PY_LINT) $(EXAMPLES_DIR)

license-check: ## Check SPDX copyright and license information with reuse
	$(PY_LICENSE_CHECK)

compile: ## Compile Python files to catch syntax errors
	$(PY_COMPILE) $(PACKAGE_DIR)
	$(PY_COMPILE) $(TESTS_DIR)
	$(PY_COMPILE) $(EXAMPLES_DIR)

# Test targets ----------------------------------------------------------------

.PHONY: test testq testv testvv

test: compile ## Run pytest (select with TESTS="<files or node ids>")
	@$(PY_TEST) $(TESTS)

testq: compile ## Run pytest quietly
	@$(PY_TEST) --quiet $(TESTS)

testv: compile ## Run pytest verbosely
	@$(PY_TEST) --verbose $(TESTS)

testvv: compile ## Run pytest verbosely and show captured output
	@$(PY_TEST) --verbose --capture=no $(TESTS)

# Documentation targets -------------------------------------------------------

.PHONY: docs publish serve

docs: ## Build the documentation site
	$(DOCS_BUILD)

publish: docs ## Build and rsync the documentation site to the remote host
	@printf "\n"
	$(RSYNC) --delete $(SITE_DIR)/ $(SITE_DEST)/
	@printf "\n"

serve: ## Serve the documentation site locally
	$(DOCS_SERVE)

# Versioning targets ----------------------------------------------------------

.PHONY: version version-patch version-minor version-major

version: ## Print the current package version
	@$(PY_VERSION_READ)

version-patch: ## Bump the patch version and sync the citation file
	$(PY_VERSION) --bump patch
	@$(MAKE) --no-print-directory citation

version-minor: ## Bump the minor version and sync the citation file
	$(PY_VERSION) --bump minor
	@$(MAKE) --no-print-directory citation

version-major: ## Bump the major version and sync the citation file
	$(PY_VERSION) --bump major
	@$(MAKE) --no-print-directory citation

.PHONY: citation

# Releasing targets -----------------------------------------------------------

# sync the version and release date into the citation metadata
citation: ## Sync version and release date into CITATION.cff
	@version=$$($(PY_VERSION_READ)); \
	date=$$(date -u +%F); \
	$(SED) -i "s|^version: .*|version: \"$$version\"|" $(CITATION_FILE); \
	$(SED) -i "s|^date-released: .*|date-released: \"$$date\"|" $(CITATION_FILE); \
	echo "updated $(CITATION_FILE): version $$version, date-released $$date"

.PHONY: release-body

# print the release notes of the current version without frontmatter and title,
# which is the body of the GitHub release; match the full title, since the SPDX
# header in the frontmatter also starts with "# "
release-body: ## Print the current version's release notes for the GitHub release
	@version=$$($(PY_VERSION_READ)); \
	file=$(RELEASES_DIR)/v$$version.md; \
	$(TEST) -f $$file || { echo "missing release notes: $$file" >&2; exit 1; }; \
	$(SED) '1,/^# Release v/d' $$file

# Cleaning targets ------------------------------------------------------------

.PHONY: clean distclean

clean: ## Remove the built site, build output, and Python caches
	$(RM) $(SITE_DIR) dist .pytest_cache
	find $(PACKAGE_DIR) $(TESTS_DIR) $(EXAMPLES_DIR) -type d -name __pycache__ -prune -exec rm -rf {} +

distclean: clean ## Remove everything clean does, plus the virtual environment
	$(RM) .venv .basedpyright *.egg-info
