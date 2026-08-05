---
Title: Releasing a New Version to PyPI and Zenodo
Author: Johann Rudi
Co-Authored-By: Oz
Date: 2026-08-05
tags:
  - release
  - tooling
---

# Releasing a New Version to PyPI and Zenodo

A release here is a chain of four artifacts produced from one commit: 

1. a bumped version in `pyproject.toml`, 
2. a git tag, 
3. a GitHub release, and 
4. the distributions uploaded to PyPI by `.github/workflows/publish.yml`. 

The upload uses **trusted publishing**, an OpenID Connect exchange in which GitHub proves the workflow's identity to PyPI, so no API token exists anywhere in the repository or in the repository secrets. Zenodo watches the same GitHub release and archives a snapshot of the source tree under a DOI.

The step that surprises people is the trigger. Pushing a tag publishes nothing. The publish workflow listens only for a **published** GitHub release, which means a draft release sits inert until you publish it. Everything before that point is reversible, and everything after it is permanent, because PyPI refuses to reuse a filename and Zenodo cannot delete a record.

!!! note

    Two constraints the tooling enforces for you. `[tool.uv] required-version` in `pyproject.toml` pins the supported `uv` range, and a `uv` outside it refuses to run any command here. The build job also compares the tag against `project.version` and fails before building if they disagree, which is what catches a mistyped tag.

## Verifying the working tree

### Step 1: Run the checks that CI will run

Format first: CI runs `format-check`, which fails on any diff it finds. Then run the three read-only checks in one invocation.

```sh
make format
make format-check lint test
```

The test target compiles the package first, so a syntax error surfaces before pytest starts. Expect a clean formatting report, a silent `basedpyright` pass, and a pytest summary with no failures:

```text
================== <N> passed, 1 warning in <T>s ==================
```

One warning is normal: the profiler test reports that `torch.profiler` clears events at the end of each cycle. Read any other warning before you continue.

### Step 2: Confirm the lock file is current

CI sets `UV_LOCKED=1`, so a lock file that disagrees with `pyproject.toml` fails every job. Check it before you spend a release on the discovery.

```sh
uv lock --check
```

A current lock prints a resolution line and exits zero:

```text
Resolved <N> packages in <T>ms
```

A stale one names the fix:

```text
error: The lockfile at `uv.lock` needs to be updated, but `--check` was provided.

hint: To update the lockfile, run `uv lock`.
```

Run `uv lock` and commit the result as an ordinary change before starting the release. Keeping the lock update out of the version commit makes the release diff easy to read.

---

## Bumping the version

### Step 3: Preview the new version number

`--dry-run` reports the transition without touching a file, which is the cheapest way to confirm you picked the right component.

```sh
uv version --bump patch --dry-run
```

```text
deep-learning-toolkit 0.4.1 => 0.4.2
```

### Step 4: Bump the version

Pick the target by what changed for people importing `dlk`. Use `make version-patch` for bug fixes and internal changes, `make version-minor` for new public API, and `make version-major` for a breaking change to existing API.

```sh
make version-patch
```

The target bumps `project.version` and then synchronizes the citation metadata, so the tail of the output names both effects:

```text
deep-learning-toolkit 0.4.1 => 0.4.2
updated CITATION.cff: version 0.4.2, date-released 2026-08-05
```

`date-released` is set to today's UTC date. If you bump today and release next week, edit that field by hand or re-run `make citation` on the day you publish, because the field records the release date rather than the bump date.

!!! note

    `dlk/__init__.py` is absent from that list on purpose. `__version__` is read from the installed package metadata through `importlib.metadata`, so `pyproject.toml` is the only place a version literal lives in the source tree. Adding one back to `__init__.py` reintroduces a value that can drift.

### Step 5: Review and commit three files

Confirm the bump touched exactly what it should.

```sh
git status --short
git --no-pager diff
```

Three files change, and a fourth would be a red flag:

- `pyproject.toml`, where `project.version` moves.
- `uv.lock`, where the root package entry moves. `uv version` locks and syncs by default, which is why this stays consistent without a separate `uv lock` run.
- `CITATION.cff`, where `version` and `date-released` move. The `cff-version` line must be untouched; the synchronizing `sed` is anchored to avoid it.

Validate the citation file before committing, since an invalid one breaks GitHub's citation widget quietly.

```sh
uvx cffconvert --validate
```

```text
Citation metadata are valid according to schema version 1.2.0.
```

Commit the three together, using a consistent message convention so the history stays greppable.

```sh
git add pyproject.toml uv.lock CITATION.cff
git commit -m "Bump version 0.4.1 -> 0.4.2"
```

Push the branch and **let CI confirm the bumped state is green** before you tag anything. A tag pointing at a commit that fails CI is more annoying to withdraw than to avoid.

---

## Releasing on GitHub

### Step 6: Tag the release commit

Derive the tag from the version you just committed instead of retyping it. `uv version --short` prints the bare version, so one assignment gives you a value to reuse for the rest of the release.

```sh
tag="v$(uv version --short)"
git tag -a "$tag" -m "Release $tag"
git push origin "$tag"
```

Keep that shell open, because every remaining step refers to `$tag`. Deriving it once removes the whole class of mistake where the tag, the release, and `pyproject.toml` disagree by a digit.

The convention is `v` followed by the version. The build job's guard strips an optional leading `v` before comparing, so a bare version passes too; use the prefix anyway, since it keeps the release listing readable and sorts with the historical tags. Pushing the tag starts no workflow, which gives you a checkpoint: the tag exists, nothing has been published, and `git push --delete origin "$tag"` still undoes it cleanly.

### Step 7: Draft the release notes

The notes take a two-part shape: a `## Summary` in prose for readers who will never open the commit log, then a `## Changelog` of one-line entries grouped by kind. Start by listing the commits since the previous tag.

```sh
prev="$(git describe --tags --abbrev=0 "$tag^")"
git --no-pager log --no-merges --pretty=format:'%h %s' "$prev".."$tag"
```

```text
89850d0 Bump version 0.4.0 -> 0.4.1
614c019 opt: Make the type hint for validation callback functions more flexible.
26e7895 New directory for evaluation/prediction from trained models. Add eval for diffusion models.
18a5cea Improve comments and error messages.
```

`git describe --tags --abbrev=0 "$tag^"` walks back from the commit before the tag to the nearest earlier tag, so the range never needs a hardcoded version. Hand it to an agent with a prompt that fixes the format, so successive releases read alike:

```text
Write release notes for tag $tag of this repository.

Read the commits with:
  git log --no-merges --pretty=format:'%h %s%n%b' $prev..$tag

Produce exactly two sections and nothing else.

## Summary

One to three short paragraphs of prose, opening with "This version ". Describe
user-visible effects for someone who imports `dlk` and never reads commits.
Name the modules or workflows affected. Leave out file-level detail.

## Changelog

Bullets grouped under "### Features", "### Bug fixes", "### Refactorings", and
"### Documentation". Omit any group with no entries. Each bullet reads:

  - <short sha> <scope>: <lowercase description>

where <scope> is the affected module (opt, nets, loss, metrics, mgmt, eval) or
area (build, ci, docs), several allowed comma-separated. Take the scope from the
commit subject prefix when it has one. Append "(#N)" when a commit references an
issue or pull request.

Drop version-bump commits. Fold commits that formed one logical change into a
single bullet, keeping the sha of the last one.
```

Read the result before publishing it. An agent can group and phrase the entries, and it cannot know which of them a user actually cares about, so the `## Summary` is the part worth rewriting by hand.

```sh
$EDITOR RELEASE_NOTES.md
```

Keep `RELEASE_NOTES.md` untracked and delete it once the release is out, or add it to `.gitignore` if this becomes routine.

### Step 8: Create the GitHub release

!!! warning

    Publishing is the irreversible step.

Create the release from the existing tag, attaching the notes from Step 7.

```sh
gh release create "$tag" --verify-tag --notes-file RELEASE_NOTES.md
```

The command prints the URL of the new release. `--verify-tag` makes it fail if the tag is missing instead of creating one silently, which matters because a tag created by `gh` points at whatever the default branch currently is.

To see the rendered result before anything publishes, add `--draft`, check it on the web, then press the publish button. A draft fires no events at all, so neither the workflow nor Zenodo reacts until you publish. Passing `--generate-notes` in place of `--notes-file` falls back to GitHub's own list of commits and merged pull requests, which makes a serviceable seed for the agent in Step 7.

### Step 9: Watch the publish workflow and verify the result

The `release: published` event starts `publish.yml`. Follow it from the terminal.

```sh
gh run list --workflow=publish.yml --limit 1
gh run watch
```

The run has two jobs. `Build distribution` checks the tag against the version, runs `uv build --no-sources`, and uploads a `dist` artifact. `Publish to PyPI` downloads that artifact and runs `uv publish --trusted-publishing always`. The second job targets the `pypi` deployment environment, so if you have configured required reviewers on that environment, the run pauses there until someone approves it.

Confirm the published package installs and imports from a clean environment, which exercises the real wheel rather than your editable checkout.

```sh
uv run --with deep-learning-toolkit --no-project --refresh-package deep-learning-toolkit python -c "import dlk; print(dlk.__version__)"
```

`--no-project` keeps uv from installing the local source tree, and `--refresh-package` bypasses a cached older wheel. Then check the Zenodo record: a new version should appear under the existing concept DOI, carrying the metadata from `.zenodo.json`.

---

## Rehearsing a release locally

Optional; read this before a release that changes packaging, licensing, or project metadata. The same backend that CI uses is bundled in the `uv` binary, so a local build produces the same artifacts.

```sh
uv build --no-sources
```

```text
Building source distribution...
Building wheel from source distribution...
Successfully built dist/deep_learning_toolkit-0.4.1.tar.gz
Successfully built dist/deep_learning_toolkit-0.4.1-py3-none-any.whl
```

`--no-sources` disables `tool.uv.sources` resolution so the build matches what other frontends see. Inspect the wheel when metadata is what you changed:

```sh
uv run python -m zipfile --list dist/*.whl
```

The wheel should contain the `dlk` tree, a `dist-info/METADATA` whose `License-Expression` matches `project.license`, and the license text at `dist-info/licenses/LICENSE`. It should not contain `tests/`, which the single-module layout excludes structurally.

`uv publish --dry-run` exists to walk the upload path without transferring files. It still resolves credentials, so outside a workflow with the trusted publisher available it tells you little.

---

## Recovering from a failed release

Optional; read this when a run went red. What to do depends on whether anything reached PyPI.

**The build job failed.** Nothing was published. Fix the cause, then move the tag: `git tag -d "$tag"`, `git push --delete origin "$tag"`, delete the release with `gh release delete "$tag"`, and start again from Step 6. A tag guard failure means the tag and `project.version` disagree, so check which one is wrong before re-tagging.

**The publish job failed partway.** Re-run it with `gh run rerun --failed`. Re-uploading is safe against PyPI, which ignores files identical to ones it already holds, so a partial upload completes rather than colliding.

**The upload succeeded and the artifact is wrong.** The version is spent. PyPI blocks reuse of a filename even after you delete the release, so bump to the next patch version and release again. Yanking the bad version on PyPI hides it from resolvers while leaving it installable by exact pin.

**Zenodo archived the wrong metadata.** Edit the record's metadata on Zenodo, which is permitted after publication and leaves the DOI intact. Records themselves cannot be deleted, and files on a published record cannot be replaced.

---

## Things worth knowing

**Where the release is cut from.** The publish workflow reacts to the release event regardless of which commit the tag names, so it will happily publish from any branch. Decide once which branch releases come off and merge into it before tagging; nothing in the tooling enforces it.

**Two workflows, two triggers.** `ci.yml` runs on pushes to the long-lived branches and on pull requests to any branch. `publish.yml` runs on published releases only. Adding a tag-push trigger to `publish.yml` would double-fire on every release and attempt a redundant upload.

**Attestations are not generated.** `uv publish` uploads PEP 740 attestation files that already sit beside the distributions, and it does not create them. Distributions therefore ship without provenance attestations; generating them would take an extra step ahead of `uv publish`.

**Zenodo ignores `CITATION.cff`.** As long as `.zenodo.json` exists, Zenodo reads it and disregards `CITATION.cff` entirely. The two files can drift apart without any warning, so update title, authors, and license in both. `CITATION.cff` serves GitHub's "Cite this repository" widget and `cffconvert` consumers.

**The license identifier is stated in three places.** `pyproject.toml`, `CITATION.cff`, and  `.zenodo.json`. A license change therefore means editing all three files, and an invalid value in `.zenodo.json` aborts the archiving for that release, which you find out only after the release is public.

**Changing your mind before you tag.** To undo an unpushed bump, `git restore pyproject.toml uv.lock CITATION.cff` if the commit has not happened, or `git reset --soft HEAD~1` and restore if it has. Re-run `make citation` afterwards if `CITATION.cff` ended up out of step with `pyproject.toml`; the target reads the version from `pyproject.toml`, so it always converges on whatever that file now says.
