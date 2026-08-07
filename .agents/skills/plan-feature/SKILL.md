---
name: plan-feature
description: Plan a new feature for this codebase and write the plan as a dated document in docs/features/. Use when the user asks to plan, design, or scope a feature or extension (e.g. "let's plan X", "write a plan for Y", "we will work on extending Z").
---

# Feature planning workflow

Produce a concise, executable plan document in `docs/features/`, following the process that worked for the DDP feature (`2026.02__ddp__1-plan.md`, whose Section 14 addendum records what this skill's improvements come from).

## Target model tier

> **Large model.** Claude Opus / flagship GPT / Kimi K2 class

Nothing here is mechanical. Step 1 asks for gaps, meaning the operation a module lacks, which is harder to see than the one it has. Step 2 asks for a core principle that collapses many small decisions into one. Both are synthesis, and both fail quietly: a plan naming a generic principle and listing no gaps reads like a plan.

Nothing checks the result at the time it is written. Before implementing a plan, confirm the `file:line` anchors still resolve and that each claimed gap is real.

## 1. Gather (do this in parallel, before designing anything)

- **Code to be extended**: build a file-by-file inventory with `file:line` touch points. Look specifically for: shared utilities vs duplicated scaffolds (duplication means changes must be mirrored), Protocols or signatures that mirror other functions (lockstep invariants), and *gaps* — missing inverse operations (e.g. save without load), missing tests, unused hooks. Gaps become plan items.
- **With a user-provided consuming application** (e.g. `../neural_nets/mops_gan/`): find every call site into the toolkit and the arguments passed. This defines the migration contract — the plan must show what "small and obvious changes" look like on the caller's side.
- **External research** when a technology choice exists: prefer the option upstream actively maintains; capture the conclusion and why in the plan's Context (one short paragraph, not a survey).
- **Repo conventions**: docs frontmatter (`Title/Author/Co-Authored-By/Date`), test style in `tests/`, the quality gate (`make format`, `make lint`, `make test`), dependency version ranges in `pyproject.toml`, and `uv run` for all Python invocations.

## 2. Decide

- Separate decisions that are genuinely the user's (scope, tradeoffs, supported platforms) from ones derivable from the code. Ask the user's decisions early with, 2–4 questions, each with a recommended default marked "(Recommended)". Record the answers in the plan as "Settled design decisions".
- Find and state a **core principle** that collapses many small decisions into one, for example, "zero signature changes; behavior auto-detected; existing usage provably unchanged". Derive its consequences explicitly (what therefore needs *no* changes).
- Mid-session scope additions from the user are normal; fold them into the plan as first-class sections, not footnotes.

## 3. Write the plan document

**Target file**: use the exact filename the user gives, verbatim — even if the date part looks inconsistent. Otherwise follow `docs/features/YYYY.ZZ__topic__N-kind.md` (example kinds relevant to this skill are plan and usage). Reserve `N+1` for a companion usage doc.

**Structure**: (keep it scannable; only the recommended approach, no alternative surveys)

1. Frontmatter: `Title/Author/Co-Authored-By/Date`.
2. **Context** — why the change, current state including gaps, research conclusion, settled design decisions, core principle.
3. **Terminology** — short explanations of every piece of jargon the plan introduces. Write for a reader who has not used the technology.
4. **Numbered per-file sections** — for edits to existing files, use a table of pre-change `file:line` targets (label them "pre-change lines"; they shift during implementation). Call out required correctness fixes separately from mechanical changes.
5. **Application migration sketch** — a short code diff showing the consumer-side changes.
6. **Tests** — always include: (a) *baseline regression tests written and green before any edits*, freezing current numeric behavior with fixed seeds; give them a `__main__` entry point that regenerates the frozen constants, and note the dependency version they were generated with; (b) a dedicated test for the single riskiest path, named as the ship gate.
7. **Documentation** — the plan itself plus the `N+1` usage doc.
8. **Unrelated improvements observed** — report-only list of cleanups noticed during exploration; never apply them silently.
9. **Ordered implementation steps** — step 1 is writing the plan doc; regression tests come before code edits; out-of-repo follow-ups go last and are marked as such.
10. **Verification** — the full quality gate *plus* a real end-to-end smoke run (actual launcher/entry point, not only unit tests); list manual checks that need hardware the dev machine lacks as explicitly pending.
11. **Risks / edge cases**.

**Writing style**:

- Write paragraphs, list items, etc. in one line; don't use line breaks.
- Write straightforward sentences; avoid "not X, but rather Y" formulations.
- Reduce the use of n-dashes and m-dashes as much as possible. Only use them when they add a very important contrast or deeper details.

## 4. After implementation: Addendum

Append a final section "Addendum: deviations and findings during implementation" to the same plan doc. Record: API deprecations discovered (and the compat shims they forced), restructurings demanded by the type checker, version-pinned assumptions, test-infrastructure workarounds, the verification outcomes with test counts, and what remains pending. Plan for this section's existence from the start — it is what makes the document trustworthy later.

## Pitfalls learned

- Line numbers in the plan go stale the moment edits start; that is fine — they should only be search anchors, not addresses. Additional information in the text should be given so the anchors can be found after the implementation.
- Deprecation warnings during the smoke run are findings, not noise: resolve them with version-compatible shims when the dependency range spans the deprecation.
- Frozen regression constants are machine/version-specific; the regeneration entry point is mandatory, not optional.
- When tests spawn worker processes, force them onto the hardware-independent path (e.g. hide GPUs) so the suite passes on any machine.
