#!/usr/bin/env bash
#
# Scaffold the release notes file, then check it once it is written.
#
# Usage:
#   draft_release_notes.sh                # write the draft for the detected version
#   draft_release_notes.sh <tag>          # write the draft for a version already tagged
#   draft_release_notes.sh --force        # overwrite a draft that already exists
#   draft_release_notes.sh --check [<tag>]  # verify the file that was written
#
# Writing the draft fills in every mechanical part: frontmatter, the file name,
# the date, the author, one bullet per commit with its scope, and the changed
# files histograms. What is left is prose and judgement, marked TODO.
#
# The commits are printed with their bodies so they can be read while writing
# the Summary. The draft itself keeps only the subjects.
#
# Example, "draft_release_notes.sh v0.4.1":
#
#   wrote docs/releases/v0.4.1.md
#
#     mode:     backfill
#     version:  0.4.1
#     date:     2026-06-01
#     range:    v0.4.0..v0.4.1
#
#   Next: read the commits below, write the Summary, sort the Changelog bullets,
#   delete every TODO line, then run:
#
#     draft_release_notes.sh --check v0.4.1
#
#   commits in range, newest first:
#
#   89850d0 Bump version 0.4.0 -> 0.4.1
#   ...
#
# The Changelog it writes holds one bullet per commit, version bumps dropped and
# scopes filled in, under a scaffold heading the writer removes:
#
#   ### Unsorted
#
#   - 614c019 opt: make the type hint for validation callback functions more flexible
#   - 26e7895 new directory for evaluation/prediction from trained models. Add eval for diffusion models
#   - 18a5cea opt, mgmt: improve comments and error messages
#
# That middle bullet shows what the scaffold cannot do: a subject holding two
# sentences stays long, and no scope fits a commit touching three areas. Both are
# for the writer to fix.
#
# Example, "draft_release_notes.sh --check v0.4.1", run on that untouched draft:
#
#   checking docs/releases/v0.4.1.md
#
#   PASS  Date is 2026-06-01
#   PASS  title names v0.4.1
#   FAIL  TODO markers are left: 9 line(s)
#   FAIL  the Unsorted scaffold heading is still there; sort the bullets into groups
#   FAIL  Summary must open with 'This version '
#   PASS  changed files section matches a fresh render
#   PASS  all 3 commits are cited
#
#   3 check(s) failed
#
# The check exits nonzero while any FAIL stands. The changed-files section is
# compared against a fresh render line by line, so a bar retyped rather than
# pasted is caught even though it changes no heading and no number:
#
#   FAIL  changed files section differs from a fresh render, so a number or a bar was altered
#           4c4
#           < dlk/nets/conv1d.py  732  <span ...>██████</span><span ...>░░░░░░░░░░░░░</span>
#           ---
#           > dlk/nets/conv1d.py  732  <span ...>██████</span><span ...>░░░░░░░░░░░</span>
#
# A commit the file never cites is reported as INFO rather than FAIL, because
# folding several commits into one bullet is expected:
#
#   INFO  2 of 3 commits cited. Not cited: 18a5cea

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mode="write"
force=0
while [ $# -gt 0 ]; do
    case "$1" in
        --check)
            mode="check"
            shift
            ;;
        --force)
            force=1
            shift
            ;;
        *)
            break
            ;;
    esac
done

cd "$(git rev-parse --show-toplevel)"

# one source of truth for which version and range are in play
eval "$("$script_dir/release_range.sh" --facts "$@")"

# subject prefixes that state a kind of change rather than a place in the tree.
# "Fix: ..." says nothing about scope, so the scope comes from the paths instead.
TYPE_WORDS="fix|fixes|feat|feature|refactor|chore|test|tests|perf|style"

# the scope for a commit: its subject prefix when that prefix names a place,
# otherwise the area of the tree it touched
commit_scope() {
    local sha="$1" subject="$2" prefix
    if [[ "$subject" =~ ^([A-Za-z][A-Za-z0-9_-]*(,[[:space:]]*[A-Za-z][A-Za-z0-9_-]*)*):[[:space:]] ]]; then
        prefix="$(tr '[:upper:]' '[:lower:]' <<<"${BASH_REMATCH[1]}")"
        if [[ ! "$prefix" =~ ^($TYPE_WORDS)$ ]]; then
            echo "$prefix"
            return
        fi
    fi
    git show --name-only --pretty=format: "$sha" | awk '
        NF == 0 { next }
        /^dlk\// {
            n = split($0, part, "/")
            scope[n > 2 ? part[2] : "dlk"] = 1
            next
        }
        /^tests\// { scope["tests"] = 1; next }
        /^docs\// || /\.md$/ { scope["docs"] = 1; next }
        /^\.github\// { scope["ci"] = 1; next }
        { scope["build"] = 1 }
        END {
            n = 0
            for (s in scope) { n++; out = (n == 1) ? s : out ", " s }
            # more than two areas is a commit no single scope describes
            if (n > 0 && n <= 2) print out
        }
    '
}

# subject without its scope prefix, lowercased and with no trailing period
commit_description() {
    local subject="$1"
    subject="$(sed -E 's/^[A-Za-z][A-Za-z0-9_-]*(,[[:space:]]*[A-Za-z][A-Za-z0-9_-]*)*:[[:space:]]*//' <<<"$subject")"
    subject="${subject%.}"
    printf '%s%s' "$(tr '[:upper:]' '[:lower:]' <<<"${subject:0:1}")" "${subject:1}"
}

# every commit in range except the version bumps, which never earn a bullet
commit_shas() {
    git log --no-merges --pretty=format:'%h %s' "$LOG_RANGE" |
        grep -viE '^[0-9a-f]+ bump version' |
        awk '{print $1}'
}

if [ "$mode" = "check" ]; then
    problems=0
    report() { printf '%-5s %s\n' "$1" "$2"; [ "$1" = "FAIL" ] && problems=$((problems + 1)) || true; }

    if [ ! -f "$FILE" ]; then
        echo "FAIL  $FILE does not exist. Run this script with no argument first."
        exit 1
    fi
    echo "checking $FILE"
    echo

    grep -q "^Date: $DATE\$" "$FILE" &&
        report PASS "Date is $DATE" ||
        report FAIL "Date must be $DATE, the day $( [ "$MODE" = backfill ] && echo "$PREVIOUS was tagged" || echo "the version is released" )"

    grep -q "^# Release v$VERSION\$" "$FILE" &&
        report PASS "title names v$VERSION" ||
        report FAIL "title must be '# Release v$VERSION'"

    if grep -qi 'TODO' "$FILE"; then
        report FAIL "TODO markers are left: $(grep -ci 'TODO' "$FILE") line(s)"
    else
        report PASS "no TODO markers left"
    fi

    if grep -q '^### Unsorted' "$FILE"; then
        report FAIL "the Unsorted scaffold heading is still there; sort the bullets into groups"
    else
        report PASS "no scaffold heading left"
    fi

    summary="$(awk '/^## Summary$/{flag=1; next} /^## /{flag=0} flag' "$FILE" | tr -d '[:space:]')"
    if [ -z "$summary" ]; then
        report FAIL "Summary is empty"
    elif ! awk '/^## Summary$/{flag=1; next} /^## /{flag=0} flag' "$FILE" | grep -q '^This version '; then
        report FAIL "Summary must open with 'This version '"
    else
        report PASS "Summary opens with 'This version '"
    fi

    # the whole section has to be what the script renders, bars included. A bar
    # retyped instead of pasted loses or gains a block without changing anything
    # a heading check would notice, so compare the text itself.
    if ! grep -q '^## Changed files$' "$FILE"; then
        report FAIL "no '## Changed files' section"
    else
        rendered="$("$script_dir/diffstat_histogram.sh" "$RANGE" | sed '/^$/d')"
        pasted="$(awk '/^## Changed files$/ {flag = 1; next} /^## / {flag = 0} flag' "$FILE" | sed '/^$/d')"
        if [ "$pasted" = "$rendered" ]; then
            report PASS "changed files section matches a fresh render"
        else
            report FAIL "changed files section differs from a fresh render, so a number or a bar was altered"
            diff <(printf '%s\n' "$pasted") <(printf '%s\n' "$rendered") |
                head -8 | sed 's/^/        /' || true
        fi
    fi

    # folding several commits into one bullet is expected, so this only informs
    total=0
    missing=""
    for sha in $(commit_shas); do
        total=$((total + 1))
        grep -q "$sha" "$FILE" || missing="$missing $sha"
    done
    if [ -n "$missing" ]; then
        report INFO "$((total - $(wc -w <<<"$missing"))) of $total commits cited. Not cited:$missing"
        report INFO "that is correct only where those commits were folded into a bullet that cites a later sha"
    else
        report PASS "all $total commits are cited"
    fi

    echo
    if [ "$problems" -gt 0 ]; then
        echo "$problems check(s) failed"
        exit 1
    fi
    echo "all checks passed"
    exit 0
fi

if [ -e "$FILE" ] && [ "$force" -eq 0 ]; then
    echo "error: $FILE already exists. Edit it, or pass --force to start over." >&2
    exit 1
fi

mkdir -p "$(dirname "$FILE")"

{
    cat <<EOF
---
Title: Release v$VERSION
Author: $AUTHOR
Co-Authored-By: TODO the model name when an agent wrote these notes, or delete this line
Date: $DATE
tags:
  - release
  - changelog
---

# Release v$VERSION

## Summary

TODO one to three short paragraphs. Open with "This version ". Describe what
TODO someone who imports dlk can now observe. Name the modules and workflows
TODO affected. Leave out file-level detail. A small release gets a short summary.

## Changelog

TODO Move every bullet below into one of these groups, in this order, and delete
TODO the groups that stay empty: Features, Bug fixes, Refactorings, Documentation,
TODO Build and CI. Write each as "### Features" and so on. Fold commits that formed
TODO one logical change into a single bullet, keeping the sha of the last one.
TODO Then delete the "### Unsorted" heading and these TODO lines.

### Unsorted

EOF

    for sha in $(commit_shas); do
        subject="$(git log -1 --pretty=format:'%s' "$sha")"
        scope="$(commit_scope "$sha" "$subject")"
        description="$(commit_description "$subject")"
        if [ -n "$scope" ]; then
            echo "- $sha $scope: $description"
        else
            echo "- $sha $description"
        fi
    done

    echo
    echo "## Changed files"
    echo
    "$script_dir/diffstat_histogram.sh" "$RANGE"
} >"$FILE"

cat <<EOF
wrote $FILE

  mode:     $MODE
  version:  $VERSION
  date:     $DATE
  range:    $RANGE

Next: read the commits below, write the Summary, sort the Changelog bullets,
delete every TODO line, then run:

  $(basename "$0") --check${1:+ $1}

EOF

echo "commits in range, newest first:"
echo
git --no-pager log --no-merges --pretty=format:'%h %s%n%b' "$LOG_RANGE"
