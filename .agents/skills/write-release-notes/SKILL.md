---
name: write-release-notes
description: Write the release notes for a version of the Deep Learning Toolkit as a changelog page in docs/releases/, which also becomes the GitHub release body. Use when the user asks for release notes, a changelog entry, or a summary of what changed in a version, and whenever a release is being prepared following docs/guides/dev/releasing.md.
---

# Writing release notes

One text with two destinations. `docs/releases/v<version>.md` is the artifact and the source of truth: a page of the documentation that stays readable years later. The GitHub release body is that same file with its frontmatter and title stripped, so the two can never disagree.

## When this runs

During a release, after `make version-*` and before the version-bump commit. The file is committed together with `pyproject.toml`, `uv.lock`, and `CITATION.cff`, so the tagged commit contains its own notes. Step 4 of `docs/guides/dev/releasing.md` is the caller.

Outside a release, the same skill backfills notes for a tag that already exists. Say which mode you are in before you start, because the commit range differs.

## 1. Establish the range

In a release, the bump is not committed yet, so the range ends at `HEAD` and contains no bump commit:

```sh
version="$(uv version --short)"
prev="$(git describe --tags --abbrev=0)"
git --no-pager log --no-merges --pretty=format:'%h %s%n%b' "$prev"..HEAD
```

Backfilling a tag walks back from the commit before it, and the range usually does contain bump commits, sometimes more than one when the tag was placed after the bump:

```sh
prev="$(git describe --tags --abbrev=0 "$tag^")"
git --no-pager log --no-merges --pretty=format:'%h %s%n%b' "$prev".."$tag"
```

The first release of a repository has no earlier tag. `git describe` fails there; list the history from the root instead with `git log --no-merges <tag>`.

## 2. Read for user-visible effects

Commit subjects are an index, not the content. Read the bodies, and for anything that moves a signature, a default, or a dependency, read the change itself with `git show --stat <sha>`. The Summary is written for someone who imports `dlk` and will never open the log: they care that parameter files now default to YAML, not that a file was renamed.

Decide which commits form one logical change while you read. A refactor spread over four commits is one bullet.

## 3. Write the file

Write `docs/releases/v<version>.md`, one file per released version:

```markdown
---
Title: Release v<version>
Author: <repository author>
Co-Authored-By: <model name, when an agent wrote the notes>
Date: <YYYY-MM-DD>
tags:
  - release
  - changelog
---

# Release v<version>

## Summary

<One to three short paragraphs of prose, opening with "This version ".>

## Changelog

### Features

- <short sha> <scope>: <lowercase description>
```

- `Date` is the day the version is released, matching `date-released` in `CITATION.cff`.
- The Summary describes user-visible effects and names the modules or workflows affected. Leave out file-level detail.
- Changelog groups appear in this order, and any group with no entries is omitted: `### Features`, `### Bug fixes`, `### Refactorings`, `### Documentation`, `### Build and CI`.
- `<scope>` comes from the commit subject prefix when it has one, otherwise from the top-level package under `dlk/` that the commit touches, otherwise from the area for repository-wide work (build, ci, docs). Several scopes are allowed, comma-separated.
- Append `(#N)` only when the commit message itself references an issue or pull request. Never infer a number.
- Drop version-bump commits.
- Fold commits that formed one logical change into a single bullet, keeping the sha of the last one.

Follow the house voice from `.agents/skills/write-user-guide/SKILL.md`: concrete nouns, no em dashes, no "not X, but Y". Descriptions after the scope start lowercase and stay on one line.

## 4. Check the result

- Every commit in the range is either in a bullet or deliberately dropped. Count them; a range of 30 commits that produced 8 bullets needs the folding to be visible, not accidental.
- The file name matches `uv version --short` exactly, including the `v` prefix on the file but not in the version itself.
- The Summary claims only things a reader can observe from outside the repository.
- Nothing in the Summary was invented to fill space. A small release gets a short summary.

## 5. Hand it to the GitHub release

The release body is the file without its frontmatter and title, which the `release-body` make target prints:

```sh
make release-body | gh release create "$tag" --verify-tag --notes-file -
```

The target resolves the file from `uv version --short`, deletes everything through the first line starting with `# `, and fails when the file is missing. GitHub already shows the tag as the release title, which is why the page title is dropped.
