---
Title: Releasing a New Version to PyPI and Zenodo
Author: Johann Rudi
Co-Authored-By: Oz
Date: 2026-08-05
tags:
  - release
  - tooling
link: "[[2026.01__build_and_tools_to_uv__1-spec]]"
---

# Releasing a New Version to PyPI and Zenodo

A release is four things produced from one commit:

1. a bumped version in `pyproject.toml`,
2. a git tag,
3. a GitHub release, and
4. the distributions uploaded to PyPI by `.github/workflows/publish.yml`.

The upload uses **[trusted publishing]**, an OpenID Connect exchange in which GitHub proves the workflow's identity to PyPI, so no API token exists anywhere in the repository or in the repository secrets. Zenodo watches the same GitHub release and archives a snapshot of the source tree under a DOI.

The step that surprises people is the trigger. Pushing a tag publishes nothing. The publish workflow listens only for a **published** GitHub release, which means a draft release sits inert until you publish it. Everything before that point is reversible, and everything after it is permanent, because PyPI refuses to reuse a filename and Zenodo cannot delete a record.

!!! note "Before you start"

    The publish workflow reacts to the release event regardless of which commit the tag names, so it will publish from any branch. Decide which branch releases come off and merge into it before you begin; nothing in the tooling enforces it. You also need `gh` authenticated against the repository, permission to push tags, and, for the archive half, the repository already enabled in Zenodo ([archiving a GitHub repository]).

    The tooling does enforce two constraints for you. `[tool.uv] required-version` in `pyproject.toml` pins the supported `uv` range, and a `uv` outside it refuses to run any command here. The build job compares the tag against `project.version` and fails before building if they disagree, which is what catches a mistyped tag.

## Verifying the working tree

### Step 1: Run the checks that CI will run

Format first: CI runs `format-check`, which fails on any diff it finds. Then run the three read-only checks in one invocation.

```sh
make format
make format-check lint test
```

The test target compiles the package first, so a syntax error surfaces before pytest starts. Expect a clean formatting report, a silent `basedpyright` pass, and a pytest summary with no failures:

```text
================== <N> passed, <W> warnings in <T>s ==================
```

The profiler test reports that `torch.profiler` clears events at the end of each cycle, and that warning is expected. Read anything else `pytest` prints before you continue.

### Step 2: Confirm the lock file is current

CI sets `UV_LOCKED=1`, so a lock file that disagrees with `pyproject.toml` fails every job. Check it before you spend a release on the discovery.

```sh
uv lock --check
```

A current lock prints a resolution line and exits zero:

```text
Resolved <N> packages in <T>ms
```

A stale one exits non-zero and names its own fix. Run `uv lock` and commit the result as an ordinary change before starting the release. Keeping the lock update out of the version commit makes the release diff easy to read.

---

## Bumping the version

### Step 3: Bump the version

Pick the target by what changed for people importing `dlk`. Use `make version-patch` for bug fixes and internal changes, `make version-minor` for new public API, and `make version-major` for a breaking change to existing API. To see the transition before anything is written, run `uv version --bump patch --dry-run` first; it reports the same line without touching a file.

Capture the outgoing version first, so the commit message in Step 4 can name both ends of the bump without you retyping either.

```sh
old="$(uv version --short)"
make version-patch
```

The target bumps `project.version` and then synchronizes the citation metadata, so the tail of the output names both effects:

```text
deep-learning-toolkit 0.4.1 => 0.4.2
updated CITATION.cff: version 0.4.2, date-released <YYYY-MM-DD>
```

`date-released` records the day the version goes out, and `make citation` fills in today's UTC date. If you bump today and release next week, re-run `make citation` on the day you publish, or edit the field by hand.

### Step 4: Review and commit three files

Confirm the bump touched exactly what it should.

```sh
git status --short
git --no-pager diff
```

Three files change, and a fourth would be a red flag:

- `pyproject.toml`, where `project.version` moves.
- `uv.lock`, where the root package entry moves. `uv version` locks and syncs by default, which is why this stays consistent without a separate `uv lock` run.
- `CITATION.cff`, where `version` and `date-released` move. The `cff-version` line must be untouched; the synchronizing `sed` is anchored to avoid it.

!!! note

    `dlk/__init__.py` is absent from that list on purpose. `__version__` is read from the installed package metadata through `importlib.metadata`, so `pyproject.toml` is the only place a version literal lives in the source tree. Adding one back to `__init__.py` reintroduces a value that can drift.

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
git commit -m "Bump version $old -> $(uv version --short)"
```

Push the branch and **let CI confirm the bumped state is green** before you tag anything. A tag pointing at a commit that fails CI is more annoying to withdraw than to avoid.

---

## Releasing on GitHub

### Step 5: Tag the release commit

Derive the tag from the version you just committed instead of retyping it. `uv version --short` prints the bare version, so one assignment gives you a value to reuse for the rest of the release.

```sh
tag="v$(uv version --short)"
git tag -a "$tag" -m "Release $tag"
git push origin "$tag"
```

Keep that shell open, because every remaining step refers to `$tag`; the same assignment reproduces it if you lose it. Deriving the tag once removes the whole class of mistake where the tag, the release, and `pyproject.toml` disagree by a digit.

The convention is `v` followed by the version. The build job's guard strips an optional leading `v` before comparing, so a bare version passes too; use the prefix anyway, since it keeps the release listing readable and sorts with the historical tags.

The pushed tag is your last cheap checkpoint. Nothing has reached PyPI yet, and `git push --delete origin "$tag"` undoes it.

### Step 6: Draft the release notes

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

where <scope> comes from the commit subject prefix when it has one, and
otherwise from the top-level package under `dlk/` that the commit touches, or
from the area for repository-wide work (build, ci, docs). Several scopes are
allowed, comma-separated. Append "(#N)" when a commit references an issue or
pull request.

Drop version-bump commits. Fold commits that formed one logical change into a
single bullet, keeping the sha of the last one.
```

Read the result before publishing it. An agent can group and phrase the entries, and it cannot know which of them a user actually cares about, so the `## Summary` is the part worth rewriting by hand.

```sh
$EDITOR RELEASE_NOTES.md
```

Keep `RELEASE_NOTES.md` untracked and delete it once the release is out, or add it to `.gitignore` if this becomes routine.

### Step 7: Create the GitHub release

!!! warning

    Publishing is the irreversible step.

Create the release from the existing tag, attaching the notes from Step 6.

```sh
gh release create "$tag" --verify-tag --notes-file RELEASE_NOTES.md
```

The command prints the URL of the new release. `--verify-tag` makes it fail if the tag is missing instead of creating one silently, which matters because a tag created by `gh` points at whatever the default branch currently is. See the [gh release create] manual for the remaining flags.

To see the rendered result before anything publishes, add `--draft`, check it on the web, then press the publish button. Passing `--generate-notes` in place of `--notes-file` falls back to GitHub's own list of commits and merged pull requests, which makes a serviceable seed for the agent in Step 6.

---

## Watching the publish workflow

Optional; read this when you want to see the upload happen instead of learning about it from a failure notification. The `release: published` event starts `publish.yml`. Follow it from the terminal.

```sh
gh run list --workflow=publish.yml --limit 1
gh run watch
```

The run builds the distributions in one job and uploads them in a second, which `.github/workflows/publish.yml` spells out. The part worth knowing before you watch it: the upload job targets the `pypi` deployment environment, so required reviewers configured on that environment hold the run until someone approves it.

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
Successfully built dist/deep_learning_toolkit-<version>.tar.gz
Successfully built dist/deep_learning_toolkit-<version>-py3-none-any.whl
```

`--no-sources` disables `tool.uv.sources` resolution so the build matches what other frontends see; the [uv packaging guide] covers the build and publish commands in full. Inspect the wheel when metadata is what you changed:

```sh
uv run python -m zipfile --list dist/*.whl
```

The wheel should contain the `dlk` tree, a `dist-info/METADATA` whose `License-Expression` matches `project.license`, and the license text at `dist-info/licenses/LICENSE`. It should not contain `tests/`, which the single-module layout excludes structurally.

`uv publish --dry-run` exists to walk the upload path without transferring files. It still resolves credentials, so outside a workflow with the trusted publisher available it tells you little.

---

## Recovering from a failed release

Optional; read this when a run went red. What to do depends on whether anything reached PyPI.

**The build job failed.** Nothing was published. Fix the cause, then move the tag: `git tag -d "$tag"`, `git push --delete origin "$tag"`, delete the release with `gh release delete "$tag"`, and start again from Step 5. A tag guard failure means the tag and `project.version` disagree, so check which one is wrong before re-tagging.

**The publish job failed partway.** Re-run it with `gh run rerun --failed`. Re-uploading is safe against PyPI, which ignores files identical to ones it already holds, so a partial upload completes rather than colliding.

**The upload succeeded and the artifact is wrong.** The version is spent. PyPI blocks reuse of a filename even after you delete the release, so bump to the next patch version and release again. Yanking the bad version on PyPI hides it from resolvers while leaving it installable by exact pin.

**Zenodo archived the wrong metadata.** Edit the record's metadata on Zenodo, which is permitted after publication and leaves the DOI intact. Records themselves cannot be deleted, and files on a published record cannot be replaced.

---

## Things worth knowing

**Two workflows, two triggers.** `ci.yml` runs on pushes to the long-lived branches and on pull requests to any branch. `publish.yml` runs on published releases only. Adding a tag-push trigger to `publish.yml` would double-fire on every release and attempt a redundant upload.

**Attestations are not generated.** As of uv 0.12, `uv publish` uploads PEP 740 attestation files that already sit beside the distributions, and it does not create them (see [uploading attestations]). Distributions therefore ship without provenance attestations; generating them would take an extra step ahead of `uv publish`.

**Zenodo ignores `CITATION.cff`.** As long as `.zenodo.json` exists, Zenodo reads it and disregards `CITATION.cff` entirely. The two files can drift apart without any warning, so update title, authors, and license in both. `CITATION.cff` serves GitHub's "Cite this repository" widget and other consumers of the [Citation File Format].

**The license identifier is stated in three places.** `pyproject.toml`, `CITATION.cff`, and `.zenodo.json`. A license change therefore means editing all three files, and an invalid value in `.zenodo.json` aborts the archiving for that release, which you find out only after the release is public.

**Changing your mind before you tag.** To undo an unpushed bump, `git restore pyproject.toml uv.lock CITATION.cff` if the commit has not happened, or `git reset --soft HEAD~1` and restore if it has. Re-run `make citation` afterwards if `CITATION.cff` ended up out of step with `pyproject.toml`; the target reads the version from `pyproject.toml`, so it always converges on whatever that file now says.


[trusted publishing]: https://docs.pypi.org/trusted-publishers/
[uv packaging guide]: https://docs.astral.sh/uv/guides/package/
[uploading attestations]: https://docs.astral.sh/uv/guides/package/#uploading-attestations-with-your-package
[gh release create]: https://cli.github.com/manual/gh_release_create
[archiving a GitHub repository]: https://help.zenodo.org/docs/github/archive-software/
[Citation File Format]: https://citation-file-format.github.io
