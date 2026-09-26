#!/usr/bin/env bash
#
# Report the state of the repository relative to a plan: the branch, the plan's
# base commit, the steps already committed, and which files the plan references
# have changed since the plan was written.
#
# The branch is compared with the plan's Branch frontmatter field, which names
# the one branch the plan is implemented on.
#
# Usage:
#   plan_status.sh <plan.md>
#
# The base commit is the plan's Base-Commit frontmatter field. If the field is
# missing, or its commit no longer exists (a rebase dropped it), the base is the
# last commit before the end of the plan's Date, marked approximate.
#
# Referenced files are the backticked tokens in the plan that name a file, at
# HEAD or at the base, with ":line" and "::test" suffixes removed. Tokens that
# name no file in the repository (new files, other repositories) are listed as
# not in repo.
#
# Drift is every commit since the base that touches a referenced file and is not
# a step of this plan. Step commits end in "(plan <id> step N)." where <id> is
# the plan's file name prefix, e.g. 2026.008. No drift means the plan's line
# numbers are valid as written.
#
# Example, "plan_status.sh docs/features/2026.008__gradient_penalties__1-plan.md":
#
#   plan:        docs/features/2026.008__gradient_penalties__1-plan.md (untracked)
#   id:          2026.008
#   branch:      feature-gp (plan has no Branch field yet)
#   base:        ebebf8d (approximate: no Base-Commit, last commit before end of 2026-09-24)
#   steps done:  none
#   files:       20 referenced, 12 in repo
#   not in repo: dlk/loss/spectral_norm.py fhn_gan/pytorch/train_gan.py ...
#   uncommitted: none
#   drift:
#   875ab42 nets: Rebuild ResidualBlock of MLPResNet after UniversalMultiLevelBlock.
#     dlk/nets/mlp.py
#     tests/nets/test_mlp.py
#   ...
#
# Without drift the last line reads "drift:       none, line numbers valid as written".

set -euo pipefail

if [ $# -ne 1 ] || [ ! -f "$1" ]; then
    echo "usage: $(basename "$0") <plan.md>" >&2
    exit 1
fi

plan="$(realpath "$1")"
cd "$(git rev-parse --show-toplevel)"
plan="$(realpath --relative-to=. "$plan")"
id="$(basename "$plan" | sed -E 's/__.*//')"

# read a key from the YAML frontmatter between the first two "---" lines
frontmatter_key() {
    awk -v key="$1" '
        NR == 1 && /^---$/ { in_fm = 1; next }
        in_fm && /^---$/ { exit }
        in_fm && $0 ~ "^" key ":" { sub("^" key ": *", ""); print; exit }
    ' "$plan"
}

if git ls-files --error-unmatch "$plan" >/dev/null 2>&1; then
    tracked="tracked"
else
    tracked="untracked"
fi

branch="$(git branch --show-current)"
plan_branch="$(frontmatter_key Branch)"
case "$branch" in
    main | next) branch_note=" (protected: create a feature branch first)" ;;
    *)
        if [ -z "$plan_branch" ]; then
            branch_note=" (plan has no Branch field yet)"
        elif [ "$branch" != "$plan_branch" ]; then
            branch_note=" (MISMATCH: plan's Branch is $plan_branch)"
        else
            branch_note=" (matches the plan's Branch)"
        fi
        ;;
esac

base="$(frontmatter_key Base-Commit)"
if [ -n "$base" ] && git cat-file -e "$base^{commit}" 2>/dev/null; then
    base_note="from Base-Commit"
else
    date="$(frontmatter_key Date)"
    if [ -n "$base" ]; then
        base_note="approximate: Base-Commit $base not found, last commit before end of $date"
    else
        base_note="approximate: no Base-Commit, last commit before end of $date"
    fi
    base="$(git log -1 --format=%H --before="$date 23:59:59" HEAD)"
fi

steps="$(git log --format=%s --fixed-strings --grep="(plan $id step " HEAD \
    | sed -nE 's/.*\(plan [^ ]+ step ([0-9]+)\)\.?$/\1/p' | sort -nu | paste -sd ' ')"

# a token names a file if it is a blob at HEAD or at the base
is_file() {
    git cat-file -e "HEAD:$1" 2>/dev/null || git cat-file -e "$base:$1" 2>/dev/null
}

referenced=()
not_in_repo=()
while IFS= read -r token; do
    [ "$token" = "$plan" ] && continue
    if is_file "$token"; then
        referenced+=("$token")
    elif [[ "$token" == */* && "$token" =~ \.[a-z]+$ ]]; then
        not_in_repo+=("$token")
    fi
done < <(grep -oE '`[^` ]+`' "$plan" | tr -d '`' \
    | sed -E 's/::.*//; s/:.*//' | grep -vE '^(\.\.|/|~)' | sort -u)

cat <<EOF
plan:        $plan ($tracked)
id:          $id
branch:      $branch$branch_note
base:        $(git rev-parse --short "$base") ($base_note)
steps done:  ${steps:-none}
files:       $((${#referenced[@]} + ${#not_in_repo[@]})) referenced, ${#referenced[@]} in repo
not in repo: ${not_in_repo[*]:-none}
EOF

if [ ${#referenced[@]} -eq 0 ]; then
    echo "uncommitted: none"
    echo "drift:       unknown, the plan references no files in repo; check anchors by search"
    exit 0
fi

dirty="$(git status --porcelain -- "${referenced[@]}")"
if [ -z "$dirty" ]; then
    echo "uncommitted: none"
else
    echo "uncommitted:"
    echo "$dirty"
fi

# list each foreign commit, then the referenced files it changed, indented
drift="$(git log --format='commit %h %s' --name-only --invert-grep --fixed-strings \
    --grep="(plan $id step " "$base..HEAD" -- "${referenced[@]}" \
    | awk 'NF == 0 { next } sub(/^commit /, "") { print; next } { print "  " $0 }')"
if [ -z "$drift" ]; then
    echo "drift:       none, line numbers valid as written"
else
    echo "drift:"
    echo "$drift"
fi
