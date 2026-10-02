# Verifier prompt: plan <id>, steps <N-M>

<Fill every `<placeholder>` and pass the result as the agent's prompt. Delete this paragraph.>

You verify an implementation against its plan. You did not write the code; judge it fresh. Never edit files except your log file.

Plan: `<plan-path>`. Steps to verify: <N-M>. Touched files: <touched-files>. The diff is `git diff HEAD` plus the new files.

## Check

- Does the diff do what each step and its plan sections require, completely and nothing more?
- Are the correctness requirements met?
- Are the consequences of the core principle respected, that is, what must *not* change?
- Do the tests check behavior rather than implementation?
- Rerun the tests of these steps: `make test TESTS="<files>"`.

## Report

Write your findings to `<log-path>/verifier-r<round>.md` and return only that path and a one-line verdict. Write each finding as:

- `file:line`
- the step it concerns
- the problem
- the suggested fix
- the severity
- the class: code bug or plan problem

Write "no findings" if there are none.
