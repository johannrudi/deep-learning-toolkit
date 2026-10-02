---
name: impl-feature
description: Implement a feature plan from docs/features/ in bundles of steps, with an implementer, a verifier, and a simplifier agent per bundle and one commit per bundle. Use when the user asks to implement, execute, or carry out a plan written with plan-feature (e.g. "implement the plan", "let's implement 2026.008", "start on the gradient penalties plan").
---

# Feature implementation workflow

Carry out the "Ordered implementation steps" of a plan written with `plan-feature`. You are the orchestrator: you brief and launch agents, decide on their findings, and commit. You do not write code or tests yourself. You may apply wording fixes to docs directly; note each in `decision-rR.md`.

## Roles

- **Implementer** (`Claude Sonnet` or `Claude Opus` or similar tiers, chosen per bundle, see "Bundle the steps"): carries out all steps of one bundle, in order, exactly as briefed. Edits code, tests, and its own log file.
- **Verifier** (`Claude Opus` or similar tier): checks every step of the bundle against the plan and the codebase. Edits no files except its own log file.
- **Simplifier** (`Claude Opus` or similar tier): suggests changes for clarity and simplicity across the bundle. Edits no files except its own log file.

A bundle is a run of consecutive steps that one implementer carries out in a single pass and that is committed once. Each bundle costs one brief, one review, and one commit, however many steps it holds.

The implementer executes, so its brief must leave no design choices open. The reviewers assess open-ended questions, and each starts fresh so it does not inherit the implementer's assumptions.

## Templates

The prompts of the three agents live in `templates/`: `implementer-brief.md`, `verifier-prompt.md`, and `simplifier-prompt.md`. Read a template when you need it, fill its `<placeholders>`, and pass the filled text as the agent's prompt. They hold the checks, the finding format, and the report format; do not restate them in SKILL.md or in the prompts.

## Log

Agents never talk to each other; every message passes through you. Log each one under `logs/impl/<id>/` (gitignored), in three layers from skim to full detail:

```
logs/impl/<id>/
  index.md              the bundling table, then one row per round: steps, round, verdicts, finding counts (kept/dropped), tokens, wall-time, commit; a total row at the end
  step-NN-MM/           one directory per bundle of steps NN to MM (`step-NN` for a bundle of one step)
    brief.md            you → implementer, verbatim
    implementer-rR.md   its report
    verifier-rR.md      its findings
    simplifier-rR.md    its findings
    decision-rR.md      you: each finding kept or dropped, with the reason; the kept list is the next fix request
    metrics.md          you: tokens and wall-time per agent run, see below
    transcripts.md      agent id → path of its raw transcript under ~/.claude/projects/
```

Each agent writes its report to its own log file and returns only the path and a one-line verdict. Read the file when you decide; this keeps earlier bundles' reports out of your context. You write `brief.md`, `decision-rR.md`, `metrics.md`, `transcripts.md`, and `index.md`. Link the raw transcripts; never copy them.

Metrics: after each agent run (implementer, fix, verifier, simplifier), take the consumed tokens and the wall-time from the agent's completion result and add a row to `metrics.md`: agent, round, model, tokens, wall-time. Write "not reported" where the result gives no figure; never estimate. End `metrics.md` with the bundle's totals and copy them into the round's row in `index.md`. The verifier and the simplifier run in parallel, so their wall-time counts once, as the longer of the two. The total row of `index.md` holds the summed tokens and the elapsed wall-time from the start of Setup to the end of Finish.

The log holds messages only. Agents put scratch files in their scratchpad. Anything a committed doc cites as evidence, such as a measurement script and its results, goes in the repository and is committed with its bundle; committed files never cite `logs/`.

## 1. Setup

- Read the plan in full, then run `.agents/skills/impl-feature/scripts/plan_status.sh <plan.md>`. It reports the branch, the base commit, the steps done, and the drift since the base.
- Drift: for each drifted file, check the plan's anchors in it by search text and confirm the gaps it claims are still real. With no drift, the line numbers are valid as written. An `approximate` base means the plan predates `Base-Commit`; say so in the addendum. Never write or change `Base-Commit`: it marks where the plan's line numbers are valid. Record the commit you start from in the addendum instead.
- Branch: a plan is implemented on exactly one branch, recorded as `Branch` in the plan's frontmatter. If the field is missing, create `feature-YYYY-NNN-<topic>` from `main` or `next`, or take the current branch if it is neither, and write the field. If the field exists and the report shows a mismatch, stop and ask. Use a worktree only if the user asks for one.
- Commit the plan doc as the plan's step 1 if it is untracked, otherwise commit the new `Branch` field alone. Skip steps marked out of repo and list them in the final report.
- Resume: `steps done` lists the steps covered by commits ending in `(plan <id> step N).` or `(plan <id> steps N-M).` Bundle the steps that have none.
- Note the time; it starts the wall-time total.

Invoking this skill grants permission to commit on the plan's `Branch` only, one commit per bundle. **Never commit on any other branch, never push, never amend.**

## 2. Bundle the steps

Before the first brief, split the remaining steps into bundles of consecutive steps, anywhere from one bundle of all of them to one bundle per step. Fewer bundles mean fewer rounds, so bundle as far as the steps allow. Cut between bundles where:

- the next step cannot be briefed without open choices until a result of the previous step is seen (a measurement, or work on hardware);
- a step runs the quality gate or a long measurement and should see a settled bundle;
- the combined diff is too large for one verifier and one simplifier to review carefully.

Also choose the implementer's model per bundle. Default to a Sonnet-tier model. Choose an Opus-tier model for open-ended numerics or algorithms, subtle correctness requirements, cross-cutting refactors, or steps that depend on each other in ways the brief cannot spell out.

Report the decision to the user before starting, as a table of bundle, steps, implementer model, and reason; do not wait for a reply. Write the same table at the top of `index.md`. Report any later change to the bundling too.

## 3. Per bundle

Run bundles in order; the next bundle starts after the previous one is committed.

1. **Brief.** Rerun `plan_status.sh`. The branch must match and `uncommitted` must be none; new drift means the user changed code mid-run, so re-anchor before briefing. Fill `templates/implementer-brief.md` and save it as `brief.md`. The brief quotes the plan, never summarizes it, and leaves no design choice open.
2. **Implement.** Launch the implementer with the chosen model and `brief.md` as its prompt.
3. **Review in parallel.** Fill `templates/verifier-prompt.md` and `templates/simplifier-prompt.md` and launch both agents with the filled text as their prompts. Pass the templates' text only; the reviewers do not read the template files. The bundle's diff is `git diff HEAD` plus the new files.
4. **Decide.** Drop findings you judge wrong and note why in `decision-rR.md`. Correctness wins where the reviewers conflict. A plan problem that touches a settled decision, the core principle, or a public signature is a blocker; any other plan problem is a deviation.
5. **Fix.** Send the kept findings to the same implementer with `SendMessage`, so it keeps its context. If the verifier had findings, rerun only the verifier. Allow at most 2 fix rounds.
6. **Record and commit.** Complete `metrics.md` and the `index.md` row. Append deviations to the plan's addendum and the simplifier's out-of-diff items to its "Unrelated improvements observed" list. Stage only the touched files and the plan doc, never `git add -A` (the repo holds untracked user files). Commit once for the bundle as `<area>: <what the bundle did> (plan <id> steps N-M).`, or `(plan <id> step N).` for a bundle of one step, where `<id>` is the plan's file name prefix (e.g. `2026.008`), with the co-author trailer, matching the `git log` style. A bundle that changes no tracked file (done upstream, or a gate run) still gets its commit, with `--allow-empty`, so `steps done` stays complete.

Steps that run the quality gate, measure, or write the usage doc and the addendum go through the same loop. For docs, the verifier checks each claim against the implemented code.

## 4. Finish

- Confirm the full quality gate is green on the last commit: `make format-check`, `make compile`, `make lint`, `make test`.
- Write the total row of `index.md`.
- Report to the user: commits by bundle, the bundling, tokens and wall-time (per bundle and total), deviations, dropped findings, pending manual checks, and out-of-repo follow-ups.

## Stop and ask only when

- the plan conflicts with the code on a settled decision, the core principle, or a public signature;
- verifier findings remain open after 2 fix rounds (the bundle is not committed; propose splitting it);
- a step needs hardware or an external repository that is not available.
