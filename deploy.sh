#!/usr/bin/env bash
# Deploy gate for MPMF. Refuses to move the deploy tag if the host contract
# test fails. This is the check that was missing in round 89d: the package lost
# get_tool_schemas/handle_tool_call and the gateway only found out on /restart.
#
#   ./deploy.sh --check-only            # run the gate against the working tree
#   ./deploy.sh deploy-<n> [<commit>]   # test, tag, then switch /root/mpmf-live
#
# Rollback: git -C /root/mpmf-live checkout --detach <previous tag>
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIVE=/root/mpmf-live
PY=/root/.hermes/hermes-agent/venv/bin/python3

run_gate() {
  local src="$1" plugin="$2"
  echo "== host contract: src=$src plugin=$plugin"
  MPMF_LIB_PATH="$src" MPMF_PLUGIN="$plugin" "$PY" "$REPO/tests/test_host_contract.py" --verbose
  echo "== end-to-end: sync_turn must grow the store"
  MPMF_LIB_PATH="$src" "$PY" "$REPO/tests/test_e2e_store_growth.py"
}

if [ "${1:-}" = "--check-only" ]; then
  run_gate "$REPO/src" "${MPMF_PLUGIN:-/root/.hermes/plugins/mpmf_plus_user/__init__.py}"
  echo "GATE OK (no tag moved)"
  exit 0
fi

TAG="${1:?usage: deploy.sh deploy-<n> [commit]}"
COMMIT="${2:-HEAD}"

# Gate runs against a detached worktree of the CANDIDATE commit, not the live one.
TMP="$(mktemp -d /tmp/mpmf-gate-XXXXXX)"
trap 'git -C "$REPO" worktree remove --force "$TMP" >/dev/null 2>&1 || rm -rf "$TMP"' EXIT
git -C "$REPO" worktree add --detach -q "$TMP" "$COMMIT"
if ! run_gate "$TMP/src" "${MPMF_PLUGIN:-/root/.hermes/plugins/mpmf_plus_user/__init__.py}"; then
  echo "DEPLOY REFUSED: host contract failed on $COMMIT"
  echo "Live unchanged; still on $(git -C "$LIVE" describe --tags 2>/dev/null || echo unknown)"
  exit 1
fi

git -C "$REPO" tag -f -a "$TAG" -m "deploy $TAG" "$COMMIT"
git -C "$LIVE" checkout -q --detach "$TAG"
echo "DEPLOYED $TAG -> $(git -C "$LIVE" log --oneline -1)"
echo "Rollback: git -C $LIVE checkout --detach <previous tag> && /restart"
