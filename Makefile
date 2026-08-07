# --------------------------------------
# Makefile for this package
#
# Author: Johann Rudi
# --------------------------------------

# set shell programs
CP     := cp
MKDIR  := mkdir -p
MV     := mv -f
RM     := rm -f
SED    := sed
TEST   := test

# set Python commands
PY_FORMAT        := uv run --no-sync black
PY_IMPORT_FORMAT := uv run --no-sync isort
PY_LINT          := uv run --no-sync basedpyright
PY_COMPILE       := uv run --no-sync python -m compileall -q -f
PY_TEST          := uv run --no-sync pytest
PY_VERSION       := uv version --no-sync
# Bare version & no color codes (e.g., for a filename):
PY_VERSION_READ  := NO_COLOR=1 $(PY_VERSION) --short

# set directories
PACKAGE_DIR  := dlk
TESTS_DIR    := tests
RELEASES_DIR := docs/releases

# set files
CITATION_FILE := CITATION.cff

# --------------------------------------

.PHONY: format format-check lint compile

format-check:
	$(PY_IMPORT_FORMAT) --check $(PACKAGE_DIR)
	$(PY_FORMAT) --check $(PACKAGE_DIR)
	@echo
	$(PY_IMPORT_FORMAT) --check $(TESTS_DIR)
	$(PY_FORMAT) --check $(TESTS_DIR)

format:
	$(PY_IMPORT_FORMAT) $(PACKAGE_DIR)
	$(PY_FORMAT) $(PACKAGE_DIR)
	@echo
	$(PY_IMPORT_FORMAT) $(TESTS_DIR)
	$(PY_FORMAT) $(TESTS_DIR)

compile:
	$(PY_COMPILE) $(PACKAGE_DIR)
	$(PY_COMPILE) $(TESTS_DIR)

lint:
	$(PY_LINT) $(PACKAGE_DIR)
	$(PY_LINT) $(TESTS_DIR)

.PHONY: test testq testv testvv

test: compile
	@$(PY_TEST)

testq: compile
	@$(PY_TEST) --quiet

testv: compile
	@$(PY_TEST) --verbose

testvv: compile
	@$(PY_TEST) --verbose --capture=no

.PHONY: version version-patch version-minor version-major

version:
	@$(PY_VERSION_READ)

version-patch:
	$(PY_VERSION) --bump patch
	@$(MAKE) --no-print-directory citation

version-minor:
	$(PY_VERSION) --bump minor
	@$(MAKE) --no-print-directory citation

version-major:
	$(PY_VERSION) --bump major
	@$(MAKE) --no-print-directory citation

.PHONY: citation

# sync the version and release date into the citation metadata
citation:
	@version=$$($(PY_VERSION_READ)); \
	date=$$(date -u +%F); \
	$(SED) -i "s|^version: .*|version: \"$$version\"|" $(CITATION_FILE); \
	$(SED) -i "s|^date-released: .*|date-released: \"$$date\"|" $(CITATION_FILE); \
	echo "updated $(CITATION_FILE): version $$version, date-released $$date"

.PHONY: release-body

# print the release notes of the current version without frontmatter and title,
# which is the body of the GitHub release
release-body:
	@version=$$($(PY_VERSION_READ)); \
	file=$(RELEASES_DIR)/v$$version.md; \
	$(TEST) -f $$file || { echo "missing release notes: $$file" >&2; exit 1; }; \
	$(SED) '1,/^# /d' $$file
