#!/usr/bin/env bash
#
# Print the changed-lines histograms for a release, ready to paste into
# docs/releases/v<version>.md.
#
# Usage:
#   diffstat_histogram.sh
#   diffstat_histogram.sh [--plain] [<range>] [-- <pathspec>...]
#
#   <range>     any git revision range, for example v0.4.1..HEAD. Left out, the
#               range comes from release_range.sh, the same one step 1 reported
#   <pathspec>  render a single histogram for these paths instead of the
#               four standard groups
#   --plain     emit a fenced code block instead of colored HTML
#
# The default output is an HTML <pre> block whose bars carry inline colors.
# A documentation site keeps those colors; GitHub strips the style attribute
# and falls back to the block characters, which carry the same information.
#
# Example, "diffstat_histogram.sh --plain v0.4.0..v0.4.1". A solid block is added
# lines, a shaded block removed lines, and the longest bar in a group is 20 units,
# so each group is scaled to its own largest file:
#
#   ### Library
#
#   ```text
#   dlk/eval/diffusion.py       199  ████████████████████
#   dlk/opt/scheduler.py         14  █░
#   dlk/opt/train_gan.py         13  █░
#   dlk/opt/train_diffusion.py   10  █░
#   dlk/mgmt/parameters.py        8  █░
#   dlk/opt/train.py              7  █░
#   dlk/opt/utils.py              7  █░
#   dlk/nets/diffusion.py         5  ░
#   dlk/__init__.py               2  █░
#   9 files, +232 -33
#   ```
#
#   ### Tests and examples
#
#   ```text
#   examples/flow_matching_2d_checkerboard.ipynb  112  ██████████░░░░░░░░░░
#   1 file, +55 -57
#   ```
#
#   ### Build and tooling
#
#   ```text
#   pyproject.toml  4  ██████████░░░░░░░░░░
#   1 file, +2 -2
#   ```
#
# A group with no changed file is left out, heading and all, which is why the
# example above has no Documentation section. Without --plain, each bar is
# wrapped in a colored span:
#
#   <pre>
#   dlk/eval/diffusion.py  199  <span style="color:#2ea043">████████████████████</span>
#   </pre>
#
# When a range touches a generated file, a footnote follows the last histogram:
#
#   Generated files, not shown above: uv.lock (+2442 -0).
#
# Examples:
#   .agents/skills/write-release-notes/scripts/diffstat_histogram.sh
#   .agents/skills/write-release-notes/scripts/diffstat_histogram.sh v0.4.1..HEAD
#   .agents/skills/write-release-notes/scripts/diffstat_histogram.sh v0.4.1..HEAD -- dlk/opt

set -euo pipefail

# Generated files, left out of the histograms and reported in a footnote instead.
# Per-file line counts say nothing about a resolved lockfile, and its size flattens
# every other bar in its group. Everything else tracked in git is shown.
GENERATED=("uv.lock")

BAR_WIDTH="${BAR_WIDTH:-20}"
ADD_COLOR="${ADD_COLOR:-#2ea043}"  # green, legible on light and dark backgrounds
DEL_COLOR="${DEL_COLOR:-#e5534b}"  # red, same

plain=0
if [ "${1:-}" = "--plain" ]; then
    plain=1
    shift
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# pathspecs below are repo-relative, so run from the top level whatever the caller's directory
cd "$(git rev-parse --show-toplevel)"

# no range given means detect it the same way step 1 does, so the two agree
if [ $# -eq 0 ] || [ "$1" = "--" ]; then
    range="$("$script_dir/release_range.sh" --range-only)"
else
    range="$1"
    shift
fi
if [ $# -gt 0 ] && [ "$1" = "--" ]; then
    shift
fi

# render one histogram for the given pathspecs, nothing if no file changed
histogram() {
    local out
    out="$(
        git diff --numstat "$range" -- "$@" |
            awk -F'\t' '{
                a = ($1 == "-") ? 0 : $1
                d = ($2 == "-") ? 0 : $2
                # collapse the "dir/{old => new}/rest" form of a rename to the new path,
                # which keeps one long rename from widening the whole column
                path = $3
                gsub(/\{[^{}]* => /, "", path)
                gsub(/\}/, "", path)
                printf "%d\t%d\t%d\t%s\n", a + d, a, d, path
            }' |
            sort -s -k1,1nr |
            awk -F'\t' \
                -v width="$BAR_WIDTH" -v plain="$plain" \
                -v add_color="$ADD_COLOR" -v del_color="$DEL_COLOR" '
                {
                    total[NR] = $1; add[NR] = $2; del[NR] = $3; path[NR] = $4
                    if ($1 > max) max = $1
                    if (length($4) > path_width) path_width = length($4)
                    if (length($1) > total_width) total_width = length($1)
                    files++; added += $2; deleted += $3
                }
                END {
                    if (files == 0) exit
                    for (i = 1; i <= files; i++) {
                        line = sprintf("%s  %*d  %s", \
                            escape(sprintf("%-*s", path_width, path[i])), \
                            total_width, total[i], \
                            bar(add[i], del[i], max, width))
                        sub(/ +$/, "", line)  # a rename with no edit has no bar
                        print line
                    }
                    printf "%d %s, +%d -%d\n", files, (files == 1 ? "file" : "files"), added, deleted
                }
                # scale a line count to bar units, never dropping a nonzero count to nothing
                function units(n, m, w,   u) {
                    if (n == 0) return 0
                    u = int(n / m * w + 0.5)
                    return (u < 1) ? 1 : u
                }
                function repeat(glyph, n,   s) {
                    s = ""
                    while (n-- > 0) s = s glyph
                    return s
                }
                # solid blocks for additions, shaded for deletions, colored unless plain
                function bar(a, d, m, w,   filled, shaded) {
                    filled = repeat("█", units(a, m, w))
                    shaded = repeat("░", units(d, m, w))
                    if (plain) return filled shaded
                    if (filled != "") filled = "<span style=\"color:" add_color "\">" filled "</span>"
                    if (shaded != "") shaded = "<span style=\"color:" del_color "\">" shaded "</span>"
                    return filled shaded
                }
                function escape(s) {
                    if (plain) return s
                    gsub(/&/, "\\&amp;", s)
                    gsub(/</, "\\&lt;", s)
                    gsub(/>/, "\\&gt;", s)
                    return s
                }
            '
    )"
    if [ -z "$out" ]; then
        return
    fi
    if [ "$plain" -eq 1 ]; then
        printf '```text\n%s\n```\n' "$out"
    else
        printf '<pre>\n%s\n</pre>\n' "$out"
    fi
}

if [ $# -gt 0 ]; then
    histogram "$@"
    exit 0
fi

# print a heading only when the group changed, so an untouched area leaves no stub
section() {
    local heading="$1" block
    shift
    block="$(histogram "$@")"
    if [ -n "$block" ]; then
        [ "$printed" -eq 1 ] && echo
        printf '%s\n\n%s\n' "$heading" "$block"
        printed=1
    fi
}

excluded=()
for path in "${GENERATED[@]}"; do
    excluded+=(":!$path")
done

printed=0
section "### Library" dlk
section "### Tests and examples" tests examples
section "### Documentation" docs '*.md'
# everything no group above claimed, so only generated files go unreported
section "### Build and tooling" . ':!dlk' ':!tests' ':!examples' ':!docs' ':!*.md' "${excluded[@]}"

# name the generated files rather than dropping them without a word
note="$(
    git diff --numstat "$range" -- "${GENERATED[@]}" |
        awk -F'\t' '{
            printf "%s%s (+%d -%d)", (NR > 1 ? ", " : ""), $3, $1, $2
        }'
)"
if [ -n "$note" ]; then
    printf '\nGenerated files, not shown above: %s.\n' "$note"
fi
