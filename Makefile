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

# set Python tools
PY_FORMAT := black
PY_IMPORT_FORMAT := isort
PY_LINT := basedpyright
PY_COMPILE := python -m compileall -q -f
PY_TEST := pytest

# set directories
PACKAGE_DIR := dlk
TESTS_DIR := tests

# --------------------------------------

.PHONY: format lint compile

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
	@$(PY_TEST) -q

testv: compile
	@$(PY_TEST) -v

testvv: compile
	@$(PY_TEST) -sv
