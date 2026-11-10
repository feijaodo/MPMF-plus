#!/usr/bin/env bash
# Install the repo's git hooks (f2: commit-msg name gate) into this clone's hooks dir.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOKS="$(git -C "$REPO" rev-parse --git-path hooks)"
case "$HOOKS" in /*) ;; *) HOOKS="$REPO/$HOOKS" ;; esac
mkdir -p "$HOOKS"
install -m 0755 "$REPO/scripts/commit-msg" "$HOOKS/commit-msg"
echo "installed commit-msg -> $HOOKS/commit-msg"
echo "needs the untracked private terms file: \${MPMF_HOME:-~/.mpmf}/private_terms.txt"
