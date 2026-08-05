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
PY_FORMAT        := uv run black
PY_IMPORT_FORMAT := uv run isort
PY_LINT          := uv run basedpyright
PY_COMPILE       := uv run python -m compileall -q -f
PY_TEST          := uv run pytest
PY_VERSION       := uv version

# set directories
PACKAGE_DIR := dlk
TESTS_DIR := tests

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

.PHONY: version version-patch version-minor version-major citation

version:
	@$(PY_VERSION) --short

version-patch:
	$(PY_VERSION) --bump patch
	@$(MAKE) --no-print-directory citation

version-minor:
	$(PY_VERSION) --bump minor
	@$(MAKE) --no-print-directory citation

version-major:
	$(PY_VERSION) --bump major
	@$(MAKE) --no-print-directory citation

# sync the version and release date into the citation metadata
citation:
	@version=$$($(PY_VERSION) --short); \
	date=$$(date -u +%F); \
	$(SED) -i "s|^version: .*|version: \"$$version\"|" $(CITATION_FILE); \
	$(SED) -i "s|^date-released: .*|date-released: \"$$date\"|" $(CITATION_FILE); \
	echo "updated $(CITATION_FILE): version $$version, date-released $$date"
