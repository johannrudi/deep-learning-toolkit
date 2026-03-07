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
PY_COMPILE := python -m py_compile
PY_TEST := pytest
PY_TESTV := pytest -v

# set package directory
PACKAGE_DIR := dlk

# --------------------------------------

.PHONY: format py_compile test

format:
	@$(FORMAT) dlk

compile:
	@printf "Run '$(PY_COMPILE)' for each '.py' file in the '$(PACKAGE_DIR)' directory\n"
	@find $(PACKAGE_DIR) -type f -name "*.py" -print0 | xargs -0 -r $(PY_COMPILE)

test: compile
	@$(PY_TEST)

testv: compile
	@$(PY_TESTV)
