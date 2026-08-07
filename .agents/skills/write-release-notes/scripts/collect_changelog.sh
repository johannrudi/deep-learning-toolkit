#!/usr/bin/env bash
#
# Collect the release notes in docs/releases into two index pages, newest
# release first:
#
#   CHANGELOG.md            the root file, read on GitHub and in Obsidian
#   docs/releases/index.md  the section landing page, for the docs site
#
# Both are written from the same source in one run, so they cannot drift apart.
# They differ only in their front matter, their title, and the depth their links
# are written from.
#
# Usage:
#   collect_changelog.sh           # rewrite both files
#   collect_changelog.sh --print   # write both to stdout instead, changing no file
#   collect_changelog.sh --check   # fail if either file is not what this renders
#
# Each entry carries the version, its date, and the Summary prose lifted from
# docs/releases/v<version>.md, followed by links to the full notes and to the
# comparison against the previous release. The per-commit changelog and the
# changed-file histograms stay in the release notes; the histograms are scaled
# per release, so stacking them here would compare bars that share no scale.
#
# Links are relative markdown, the one form that resolves on GitHub, in Obsidian,
# and in the docs site alike. Wikilinks and transclusion render only in Obsidian,
# so neither is used here.
#
# Example, one entry of CHANGELOG.md:
#
#   ## [v0.4.1](https://github.com/johannrudi/deep-learning-toolkit/releases/tag/v0.4.1) · 2026-06-01
#
#   This version separates evaluation from training. `dlk.eval` is a new module
#   for prediction and evaluation from trained models, and it starts with the
#   diffusion case, which previously had no home outside the training loop.
#
#   The type hint for validation callbacks passed into the training loops accepts
#   more shapes now, so applications whose validation function returns something
#   other than the previously declared type no longer need a cast to satisfy the
#   linter.
#
#   [Full notes](docs/releases/v0.4.1.md) ·
#   [Compare](https://github.com/johannrudi/deep-learning-toolkit/compare/v0.4.0...v0.4.1)
#
# The same entry in docs/releases/index.md differs in one link, which is written
# from inside that directory:
#
#   [Full notes](v0.4.1.md) ·
#
# The oldest release has no Compare link, having nothing to compare against.
#
# Example, "collect_changelog.sh --check" when a release was added and this was
# not rerun:
#
#   CHANGELOG.md is stale
#   docs/releases/index.md is stale
#
#   rerun: .agents/skills/write-release-notes/scripts/collect_changelog.sh
#
# The check exits nonzero while either file differs, so a release that landed
# without its index entry is caught rather than shipped.

set -euo pipefail

RELEASES_DIR="docs/releases"
CHANGELOG_FILE="CHANGELOG.md"
INDEX_FILE="$RELEASES_DIR/index.md"

# the canonical path, named in the generated files. Hard-coded rather than taken
# from $0, so the output does not change with how the script was invoked
SELF=".agents/skills/write-release-notes/scripts/collect_changelog.sh"

mode="write"
case "${1:-}" in
    --print) mode="print" ;;
    --check) mode="check" ;;
    "") ;;
    *)
        echo "error: unknown argument '$1'" >&2
        exit 1
        ;;
esac

cd "$(git rev-parse --show-toplevel)"

# the repository URL, so the links point at the right host without a network call
repo_url() {
    local url
    url="$(
        awk '
            /^\[/ { in_urls = ($0 ~ /^\[project\.urls\]/); next }
            in_urls && $0 ~ /^Repository *=/ {
                if (match($0, /"[^"]*"/)) {
                    print substr($0, RSTART + 1, RLENGTH - 2)
                    exit
                }
            }
        ' pyproject.toml
    )"
    url="${url%.git}"
    echo "${url%/}"
}

# the Date field from the front matter, not from anywhere later in the file
file_date() {
    awk '
        /^---$/ { fence++; next }
        fence == 1 && /^Date:/ {
            sub(/^Date:[[:space:]]*/, "")
            print
            exit
        }
    ' "$1"
}

# the Summary section, stripped of the blank lines that frame it
file_summary() {
    awk '
        /^## Summary$/ { flag = 1; next }
        /^## / { flag = 0 }
        flag { line[++n] = $0; if (NF) { if (!first) first = n; last = n } }
        END { for (i = first; i <= last; i++) print line[i] }
    ' "$1"
}

# the release entries, newest first. $1 is the prefix the notes are linked
# through, which differs between a file at the root and one beside the notes
entries() {
    local prefix="$1" url version date previous="" entry
    local collected=()
    url="$(repo_url)"

    # the Compare links need each release's predecessor, so walk oldest first
    # and hold the entries until the order can be reversed
    for file in $(ls "$RELEASES_DIR"/v*.md 2>/dev/null | sort -V); do
        version="$(basename "$file" .md)"
        date="$(file_date "$file")"
        entry="$(
            printf '## [%s](%s/releases/tag/%s) · %s\n\n' "$version" "$url" "$version" "$date"
            file_summary "$file"
            printf '\n[Full notes](%s%s.md)' "$prefix" "$version"
            if [ -n "$previous" ]; then
                printf ' ·\n[Compare](%s/compare/%s...%s)' "$url" "$previous" "$version"
            fi
        )"
        collected+=("$entry")
        previous="$version"
    done

    local i
    for ((i = ${#collected[@]} - 1; i >= 0; i--)); do
        printf '\n%s\n' "${collected[i]}"
    done
}

render_changelog() {
    cat <<EOF
<!-- Generated from $RELEASES_DIR by $SELF -->
<!-- Do not edit by hand; rerun that script after adding a release. -->

# Changelog

Release notes for the *Deep Learning Toolkit*, newest first. Each entry links to
its full notes, which add the per-commit changelog and the changed-file
histograms.
EOF
    entries "$RELEASES_DIR/"
}

render_index() {
    cat <<EOF
---
Title: Releases
tags:
  - release
  - changelog
---

<!-- Generated from $RELEASES_DIR by $SELF -->
<!-- Do not edit by hand; rerun that script after adding a release. -->

# Releases

Release notes for the *Deep Learning Toolkit*, newest first. Each entry links to
its full notes, which add the per-commit changelog and the changed-file
histograms.
EOF
    entries ""
}

case "$mode" in
    print)
        render_changelog
        echo
        render_index
        ;;
    check)
        stale=0
        for target in "$CHANGELOG_FILE" "$INDEX_FILE"; do
            case "$target" in
                "$CHANGELOG_FILE") rendered="$(render_changelog)" ;;
                *) rendered="$(render_index)" ;;
            esac
            if [ ! -f "$target" ]; then
                echo "$target does not exist" >&2
                stale=1
            elif ! diff -q <(printf '%s\n' "$rendered") "$target" >/dev/null; then
                echo "$target is stale" >&2
                stale=1
            fi
        done
        if [ "$stale" -eq 1 ]; then
            echo >&2
            echo "rerun: $SELF" >&2
            exit 1
        fi
        echo "$CHANGELOG_FILE and $INDEX_FILE are up to date"
        ;;
    write)
        render_changelog >"$CHANGELOG_FILE"
        render_index >"$INDEX_FILE"
        echo "wrote $CHANGELOG_FILE and $INDEX_FILE ($(ls "$RELEASES_DIR"/v*.md | wc -l) releases)"
        ;;
esac
