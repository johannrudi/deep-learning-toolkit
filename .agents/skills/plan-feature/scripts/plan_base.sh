#!/usr/bin/env bash
#
# Report the commit a plan is written against, for its Base-Commit field. The
# plan's pre-change line numbers are valid at this commit, and impl-feature
# measures drift from it.
#
# Usage:
#   plan_base.sh [<path>...]   # the files the plan edits; default: all tracked files
#
# Exits 1 if any of the paths has uncommitted changes, because the hash would
# then not describe the code the plan was written against. Commit them first.
#
# "published" lists main and next branches (local or remote) that contain the
# commit. If it is empty, the commit lives only on the current branch, and a
# rebase or squash of that branch would drop it.
#
# Example, "plan_base.sh dlk/loss/wasserstein_gan.py dlk/opt/train_gan.py":
#
#   branch:      feature-gp
#   Base-Commit: d73b7a1c0f5e8a2b9d4c6e1f3a7b8c9d0e1f2a3b
#   subject:     opt: Make generator in GAN training compatible with CUDA graphs.
#   published:   next origin/next
#   uncommitted: none
#
# Example of the refusal:
#
#   uncommitted:
#    M dlk/opt/train_gan.py
#
#   error: commit these changes before recording Base-Commit

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

head="$(git rev-parse HEAD)"
published="$(git branch -a --contains HEAD --format='%(refname:short)' \
    | grep -E '(^|/)(main|next)$' | paste -sd ' ' || true)"
if [ $# -gt 0 ]; then
    dirty="$(git status --porcelain -- "$@")"
else
    dirty="$(git status --porcelain --untracked-files=no)"
fi

cat <<EOF
branch:      $(git branch --show-current)
Base-Commit: $head
subject:     $(git log -1 --format=%s HEAD)
published:   ${published:-(none, only on this branch)}
EOF

if [ -z "$dirty" ]; then
    echo "uncommitted: none"
else
    echo "uncommitted:"
    echo "$dirty"
    echo
    echo "error: commit these changes before recording Base-Commit" >&2
    exit 1
fi
