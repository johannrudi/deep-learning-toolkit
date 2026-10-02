# Implementer brief: plan <id>, steps <N-M>

<Fill every `<placeholder>`, then save the result verbatim as `brief.md` and pass it as the agent's prompt. Delete this paragraph and any section that does not apply.>

You implement steps <N-M> of the plan `<plan-path>`, in order, exactly as written below. Do not make design choices; if the brief leaves one open or the plan cannot be followed, stop that step and report it. Do not commit.

## Steps

<for each step: its number and its text from the plan, verbatim>

## Plan sections they reference

<quoted, not summarized, with the section titles>

## Target files

<file, search anchor for the place to change; mark new files as new>

## Settled decisions and correctness requirements

<the ones that apply to these steps; include what must not change>

## Tests

<tests to write, tests to keep green>

## Commands

- `make test TESTS="<files>"`
- `make format`
- `make lint`

Run them after the last step, and after any step whose tests another step depends on. Follow the code and comment rules in `AGENTS.md`.

## Report

Write your report to `<log-path>/implementer-r<round>.md` and return only that path and a one-line verdict. For each step, report the files you touched, the test results, and any part of the plan you could not follow.
