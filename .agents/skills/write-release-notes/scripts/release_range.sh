#!/usr/bin/env bash
#
# Work out which version is being written up, which commit range it covers, and
# print that range's commits. Run it with no argument and read the output.
#
# Usage:
#   release_range.sh              # detect the version from pyproject.toml
#   release_range.sh <tag>        # write up a tag that already exists
#   release_range.sh --range-only # print just the range, for other scripts
#
# The mode is detected, not chosen. A version in pyproject.toml with no matching
# tag means a release is in progress and the range ends at HEAD. A version whose
# tag already exists means the notes are being backfilled and the range ends at
# the tag.
#
# The version is read straight out of pyproject.toml. Do not reach for
# "uv version" here: bumping with it re-locks and syncs the virtual environment,
# which replaces a torch build installed from an accelerator group.
#
# Example, "release_range.sh v0.4.1":
#
#   mode:     backfill
#   version:  0.4.1
#   file:     docs/releases/v0.4.1.md
#   date:     2026-06-01
#   author:   Johann Rudi
#   previous: v0.4.0
#   range:    v0.4.0..v0.4.1
#
#   commits (4 in range, newest first):
#
#   89850d0 Bump version 0.4.0 -> 0.4.1
#
#   614c019 opt: Make the type hint for validation callback functions more flexible.
#
#   26e7895 New directory for evaluation/prediction from trained models. Add eval
#   for diffusion models.
#
#   18a5cea Improve comments and error messages.
#
# In release mode the range ends at HEAD, the date is today, and a dirty working
# tree adds a note that the range may still grow.
#
# Example, "release_range.sh --facts v0.4.1", for use by other scripts:
#
#   MODE=backfill
#   VERSION=0.4.1
#   FILE=docs/releases/v0.4.1.md
#   DATE=2026-06-01
#   AUTHOR=Johann\ Rudi
#   PREVIOUS=v0.4.0
#   RANGE=v0.4.0..v0.4.1
#   LOG_RANGE=v0.4.0..v0.4.1
#
# Example, "release_range.sh --range-only v0.4.1":
#
#   v0.4.0..v0.4.1
#
# Example of the refusal, when pyproject.toml names an already tagged version and
# commits sit on top of it:
#
#   error: pyproject.toml says 0.4.1, tag v0.4.1 exists, and 32 commits sit on
#   top of it.
#
#   Nothing tells this script which version to write up. Pick one:
#     - releasing? run make version-patch (or version-minor, version-major) first,
#       then run this script again with no argument
#     - backfilling notes for a version already released? name it:
#       release_range.sh v0.4.1

set -euo pipefail

# the hash git gives the empty tree, used when no earlier tag exists
EMPTY_TREE="4b825dc642cb6eb9a060e54bf8d69288fbee4904"

output="human"
case "${1:-}" in
    --range-only)
        output="range"
        shift
        ;;
    --facts)
        # shell assignments, so other scripts reuse this detection instead of redoing it
        output="facts"
        shift
        ;;
esac

cd "$(git rev-parse --show-toplevel)"

# read a key from the [project] table of pyproject.toml, ignoring other tables
project_key() {
    awk -v key="$1" '
        /^\[/ { in_project = ($0 ~ /^\[project\]/); next }
        in_project && $0 ~ "^" key " *=" {
            if (match($0, /"[^"]*"/)) {
                print substr($0, RSTART + 1, RLENGTH - 2)
                exit
            }
        }
    ' pyproject.toml
}

# the first author's name, which lives in an inline table inside the authors array
project_author() {
    awk '
        /^\[/ { in_project = ($0 ~ /^\[project\]/); next }
        in_project && /^authors/ { in_authors = 1 }
        in_authors && match($0, /name *= *"[^"]*"/) {
            field = substr($0, RSTART, RLENGTH)
            match(field, /"[^"]*"/)
            print substr(field, RSTART + 1, RLENGTH - 2)
            exit
        }
        in_authors && /\]/ { in_authors = 0 }
    ' pyproject.toml
}

if [ $# -gt 0 ]; then
    # an explicit tag states the intent, so take it at face value
    tag="$1"
    version="${tag#v}"
    mode="backfill"
else
    version="$(project_key version)"
    if [ -z "$version" ]; then
        echo "error: no version found in the [project] table of pyproject.toml" >&2
        exit 1
    fi
    tag="v$version"
    if ! git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
        mode="release"
    elif [ "$(git rev-list --count "$tag..HEAD" 2>/dev/null || echo 0)" -gt 0 ]; then
        # pyproject still names a version that is already tagged, yet work sits on top
        # of it: the bump has not run, and writing up $tag would ignore all of it
        cat >&2 <<EOF
error: pyproject.toml says $version, tag $tag exists, and $(git rev-list --count "$tag..HEAD") commits sit on top of it.

Nothing tells this script which version to write up. Pick one:
  - releasing? run make version-patch (or version-minor, version-major) first,
    then run this script again with no argument
  - backfilling notes for a version already released? name it:
    $(basename "$0") $tag
EOF
        exit 1
    else
        mode="backfill"
    fi
fi

if [ "$mode" = "backfill" ]; then
    if ! git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
        echo "error: tag $tag does not exist" >&2
        exit 1
    fi
    end="$tag"
    date="$(git log -1 --format=%ad --date=short "$tag")"
else
    end="HEAD"
    date="$(date +%F)"
fi

# the previous release is the nearest tag before the end of the range
if prev="$(git describe --tags --abbrev=0 "$end^" 2>/dev/null)"; then
    log_range="$prev..$end"
    diff_range="$prev..$end"
else
    prev="(none, this is the first release)"
    log_range="$end"
    diff_range="$EMPTY_TREE..$end"
fi

if [ "$output" = "range" ]; then
    echo "$diff_range"
    exit 0
fi

if [ "$output" = "facts" ]; then
    printf 'MODE=%q\nVERSION=%q\nFILE=%q\nDATE=%q\nAUTHOR=%q\nPREVIOUS=%q\nRANGE=%q\nLOG_RANGE=%q\n' \
        "$mode" "$version" "docs/releases/v$version.md" "$date" "$(project_author)" \
        "$prev" "$diff_range" "$log_range"
    exit 0
fi

cat <<EOF
mode:     $mode
version:  $version
file:     docs/releases/v$version.md
date:     $date
author:   $(project_author)
previous: $prev
range:    $diff_range

EOF

if [ "$mode" = "release" ] && [ -n "$(git status --porcelain)" ]; then
    echo "note: the working tree is dirty, so the range may still grow before the tag"
    echo
fi

echo "commits ($(git rev-list --no-merges --count "$log_range") in range, newest first):"
echo
git --no-pager log --no-merges --pretty=format:'%h %s%n%b' "$log_range"
