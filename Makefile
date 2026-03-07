.PHONY: format py_compile test

format:
	@black dlk

py_compile:
	@find dlk -type f -name "*.py" -print0 | xargs -0 -r python -m py_compile

test:
	@pytest
