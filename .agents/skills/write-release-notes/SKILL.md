---
name: write-release-notes
description: Write the release notes for a version of the Deep Learning Toolkit as a changelog page in docs/releases/, which also becomes the GitHub release body. Use when the user asks for release notes, a changelog entry, or a summary of what changed in a version, and whenever a release is being prepared following docs/guides/dev/releasing.md.
---

# Writing release notes

One text with two destinations. `docs/releases/v<version>.md` is the artifact and the source of truth: a page of the documentation that stays readable years later. The GitHub release body is that same file with its frontmatter and title stripped, so the two can never disagree.

Scripts do the mechanical work. Every value that can be read out of the repository comes from a script, so the writing left over is prose and judgement.

## When this runs

During a release, after `make version-*` and before the version-bump commit. The file `docs/releases/v<version>.md` is committed together with `pyproject.toml`, `uv.lock`, and `CITATION.cff`, so the tagged commit contains its own notes. The guide `docs/guides/dev/releasing.md` contains a step when the present skill is used.

Outside a release, the same skill backfills notes for a tag that already exists. The commit range differs between the two, so step 1 works out which one is running.

## What you may do

- Run the scripts in `.agents/skills/write-release-notes/scripts/`.
- Run `git log`, `git show`, `git diff`, and `git status` to read history.
- Read any file in the repository.
- Create and edit one file, the `docs/releases/v<version>.md` that step 1 names.

That is the whole list. Anything else belongs to the person running the release, including committing, tagging, pushing, editing other files, and running `uv`.

## 1. Draft the file

```sh
.agents/skills/write-release-notes/scripts/draft_release_notes.sh
```

This writes the file and prints the commits in range with their bodies. It fills in the frontmatter, the file name, the date, the author, one bullet per commit with its scope, and the changed-files histograms. What it leaves are lines marked `TODO`.

Read what it prints before doing anything else. It names the version, the mode, and the range it chose.

To write up a version that is already tagged, name the tag: `draft_release_notes.sh v0.3.1`.

The script stops when `pyproject.toml` names a version whose tag already exists and commits sit on top of it, because that state does not say which version to write up. Do what the message says. Never guess a range.

It also stops when the file already exists, so a draft is never overwritten by accident.

## 2. Read the commits for user-visible effects

Commit subjects are an index. Read the bodies the script printed, and for anything that moves a signature, a default, or a dependency, read the change itself with `git show --stat <sha>`.

The Summary is written for someone who imports `dlk` and will never open the log. They care that a training loop now takes a validation callback. File renames stay out.

Decide which commits form one logical change while you read. A refactor spread over four commits is one bullet.

## 3. Write the prose

Replace every `TODO` line in the draft.

**Summary.** One to three short paragraphs, opening with "This version ". Describe user-visible effects and name the modules or workflows affected. Leave out file-level detail. A small release gets a short summary.

**Changelog.** Move each bullet out of `### Unsorted` into one of these groups, in this order, and delete the groups that stay empty:

`### Features`, `### Bug fixes`, `### Refactorings`, `### Documentation`, `### Build and CI`

Then delete the `### Unsorted` heading.

Bullets keep the shape the draft gave them, `- <short sha> <scope>: <lowercase description>`, with these corrections:

- Fold commits that formed one logical change into a single bullet, keeping the sha of the last one.
- Fill in a scope the script left blank, and fix one it guessed wrong. A scope is the top-level package under `dlk/` that the commit touches, or the area for repository-wide work (build, ci, docs). Several are allowed, comma-separated.
- Append `(#N)` only when the commit message itself references an issue or pull request. Never infer a number.

**Changed files.** Leave this section alone. Paste nothing into it and edit nothing in it. Where a renamed file matters to a reader, say the old name in the Changelog bullet, because the histogram lists a renamed file under its new path only.

Follow the voice from `.agents/skills/write-user-guide/SKILL.md`: concrete nouns, no en- and em-dashes, no "not X, but Y". Descriptions after the scope start lowercase and stay on one line, with no line wrap.

## 4. Check the result

```sh
.agents/skills/write-release-notes/scripts/draft_release_notes.sh --check
```

It verifies the date, the title, and that no `TODO` or scaffold heading survives. It also re-renders the changed-files histograms and compares them to what the file holds, line by line, so a bar that was retyped rather than pasted is caught. Fix every `FAIL` and run it again.

A line marked `INFO` reports commits the file never cites. That is correct where those commits were folded into another bullet, and a mistake where they were dropped by accident. Decide which, one sha at a time.

Then read the file once more for the things no script can see:

- The Summary claims only what a reader can observe from outside the repository.
- Nothing in the Summary was invented to fill space.
- Folding is visible. A long range that produced few bullets should read as deliberate.
