---
name: impl-feature
description: Implement a feature plan from docs/features/ step by step, with an implementer, a verifier, and a simplifier agent per step and one commit per step. Use when the user asks to implement, execute, or carry out a plan written with plan-feature (e.g. "implement the plan", "let's implement 2026.008", "start on the gradient penalties plan").
---

# Feature implementation workflow

Carry out the "Ordered implementation steps" of a plan written with `plan-feature`. You are the orchestrator: you brief and launch agents, decide on their findings, and commit. You do not write code or tests yourself. You may apply wording fixes to docs directly; note each in `decision-rR.md`.

## Roles

- **Implementer** (`Claude Sonnet` or similar): carries out one step exactly as briefed. Edits code, tests, and its own log file.
- **Verifier** (`Claude Opus` or similar): checks the step against the plan and the codebase. Edits no files except its own log file.
- **Simplifier** (`Claude Opus` or similar): suggests changes for clarity and simplicity. Edits no files except its own log file.

The implementer executes, so its brief must leave no design choices open. The reviewers assess open-ended questions, and each starts fresh so it does not inherit the implementer's assumptions.

## Log

Agents never talk to each other; every message passes through you. Log each one under `logs/impl/<id>/` (gitignored), in three layers from skim to full detail:

```
logs/impl/<id>/
  index.md              one table row per round: step, round, verdicts, finding counts (kept/dropped), commit
  step-NN/
    brief.md            you → implementer, verbatim
    implementer-rR.md   its report
    verifier-rR.md      its findings
    simplifier-rR.md    its findings
    decision-rR.md      you: each finding kept or dropped, with the reason; the kept list is the next fix request
    transcripts.md      agent id → path of its raw transcript under ~/.claude/projects/
```

Each agent writes its report to its own log file and returns only the path and a one-line verdict. Read the file when you decide; this keeps earlier steps' reports out of your context. You write `brief.md`, `decision-rR.md`, `transcripts.md`, and `index.md`. Link the raw transcripts; never copy them.

The log holds messages only. Agents put scratch files in their scratchpad. Anything a committed doc cites as evidence, such as a measurement script and its results, goes in the repository and is committed with its step; committed files never cite `logs/`.

## 1. Setup

- Read the plan in full, then run `.agents/skills/impl-feature/scripts/plan_status.sh <plan.md>`. It reports the branch, the base commit, the steps done, and the drift since the base.
- Drift: for each drifted file, check the plan's anchors in it by search text and confirm the gaps it claims are still real. With no drift, the line numbers are valid as written. An `approximate` base means the plan predates `Base-Commit`; say so in the addendum. Never write or change `Base-Commit`: it marks where the plan's line numbers are valid. Record the commit you start from in the addendum instead.
- Branch: a plan is implemented on exactly one branch, recorded as `Branch` in the plan's frontmatter. If the field is missing, create `feature-YYYY-NNN-<topic>` from `main` or `next`, or take the current branch if it is neither, and write the field. If the field exists and the report shows a mismatch, stop and ask. Use a worktree only if the user asks for one.
- Commit the plan doc as the plan's step 1 if it is untracked, otherwise commit the new `Branch` field alone. Skip steps marked out of repo and list them in the final report.
- Resume: `steps done` lists the commits ending in `(plan <id> step N).` Continue with the first step that has none.

Invoking this skill grants permission to commit on the plan's `Branch` only, one commit per step. **Never commit on any other branch, never push, never amend.**

## 2. Per step

Run steps in order. A step may start before the previous one is committed only if it reads nothing that step produces.

1. **Brief.** Rerun `plan_status.sh`. The branch must match and `uncommitted` must be none; new drift means the user changed code mid-run, so re-anchor before briefing. Write the implementer's brief: the step text, the plan sections it references (quoted, not summarized), target files with search anchors, the settled decisions and correctness requirements that apply, the tests to write or keep green, and the commands to run (`make test TESTS="<files>"`, `make format`, `make lint`). Tell it not to commit, and to report the files it touched, the test results, and any part of the plan it could not follow.
2. **Implement.** Launch the implementer.
3. **Review in parallel.** Launch the verifier and the simplifier with the plan path, the step number, and the touched files. The step's diff is `git diff HEAD` plus the new files.
   - Verifier: does the diff do what the step and its plan sections require, completely and nothing more? Check the correctness requirements, the consequences of the core principle (what must *not* change), and that tests check behavior rather than implementation. Rerun the step's tests. Classify each finding as a code bug or a plan problem.
   - Simplifier: clarity and simplicity within the diff, against the code and comment rules in `AGENTS.md` and the idiom of the surrounding code. Judge the tests as a whole too: merge near-duplicates and keep their volume in proportion to the code. Report at most 5 findings, ranked by impact; list any others in one line each as optional. List cleanups outside the diff separately, as report only.
   - Both write findings as `file:line`, problem, suggested fix, and severity, or "no findings", to their log file.
4. **Decide.** Drop findings you judge wrong and note why in `decision-rR.md`. Correctness wins where the reviewers conflict. A plan problem that touches a settled decision, the core principle, or a public signature is a blocker; any other plan problem is a deviation.
5. **Fix.** Send the kept findings to the same implementer with `SendMessage`, so it keeps its context. If the verifier had findings, rerun only the verifier. Allow at most 2 fix rounds.
6. **Record and commit.** Append deviations to the plan's addendum and the simplifier's out-of-diff items to its "Unrelated improvements observed" list. Stage only the touched files and the plan doc, never `git add -A` (the repo holds untracked user files). Commit as `<area>: <what the step did> (plan <id> step N).`, where `<id>` is the plan's file name prefix (e.g. `2026.008`), with the co-author trailer, matching the `git log` style. A step that changes no tracked file (done upstream, or a gate run) still gets its commit, with `--allow-empty`, so `steps done` stays complete.

Steps that run the quality gate, measure, or write the usage doc and the addendum go through the same loop. For docs, the verifier checks each claim against the implemented code.

## 3. Finish

- Confirm the full quality gate is green on the last commit: `make format-check`, `make compile`, `make lint`, `make test`.
- Report to the user: commits by step, deviations, dropped findings, pending manual checks, and out-of-repo follow-ups.

## Stop and ask only when

- the plan conflicts with the code on a settled decision, the core principle, or a public signature;
- verifier findings remain open after 2 fix rounds;
- a step needs hardware or an external repository that is not available.
