# --------------------------------------
# Author: Johann Rudi
# --------------------------------------

# set shell programs
CP     := cp
MKDIR  := mkdir -p
MV     := mv -f
RM     := rm -f
SED    := sed
TEST   := test

# set Python tools
PY_FORMAT := black
PY_IMPORT_FORMAT := isort
PY_LINT := basedpyright
PY_COMPILE := python -m compileall -q -f
PY_TEST := pytest
PY_TESTV := pytest -v

# set package directory
PACKAGE_DIR := dlk

# --------------------------------------

.PHONY: format lint compile test testv

format:
	$(PY_IMPORT_FORMAT) $(PACKAGE_DIR)
	$(PY_FORMAT) $(PACKAGE_DIR)

lint:
	$(PY_LINT) $(PACKAGE_DIR)

compile:
	$(PY_COMPILE) $(PACKAGE_DIR)

test: compile
	@$(PY_TEST)

testv: compile
	@$(PY_TESTV)
