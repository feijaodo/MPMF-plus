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
LIVE="${MPMF_LIVE:-/root/mpmf-live}"          # the deploy-target worktree
PY="${MPMF_PY:-/root/.hermes/hermes-agent/venv/bin/python3}"
# unattended runners may start with HOME unset; the cron dry-runs and pytest need it (set -u)
export HOME="${HOME:-$(getent passwd "$(id -u)" | cut -d: -f6)}"

# r90e: the live tree is a deploy target, never a workspace. Refuse to tag or
# test while it carries edits that exist only on this host.
if [ -n "$(git -C "$LIVE" status --porcelain 2>/dev/null)" ]; then
  echo "DEPLOY REFUSED: $LIVE has uncommitted or local-only changes:"
  git -C "$LIVE" status --short
  exit 1
fi

run_gate() {
  local src="$1" plugin="$2"
  echo "== host contract: src=$src plugin=$plugin"
  MPMF_LIB_PATH="$src" MPMF_PLUGIN="$plugin" "$PY" "$REPO/tests/test_host_contract.py" --verbose || return 1
  echo "== auto-revert: pairing, simulated drop REVERTs, pairing failure exits non-zero"
  MPMF_LIB_PATH="$src" "$PY" "$REPO/tests/test_autorevert_pairing.py" || return 1
  echo "== real turns only: cron sessions never count as the user; thinking age is relative to their last turn"
  MPMF_LIB_PATH="$src" "$PY" "$REPO/tests/test_real_turns_only.py" || return 1
  echo "== re-enrich accounting: enriched + failed + deferred == pending"
  MPMF_LIB_PATH="$src" "$PY" "$REPO/tests/test_reenrich_accounting.py" || return 1
  echo "== clean install: nightly --dry-run needs no API key"
  MPMF_LIB_PATH="$src" "$PY" "$REPO/tests/test_dry_run_no_key.py" || return 1
  echo "== end-to-end: sync_turn must grow the store"
  MPMF_LIB_PATH="$src" "$PY" "$REPO/tests/test_e2e_store_growth.py" || return 1
  echo "== name gate (f2): no private terms in tracked files / commit messages, no absolute home path in src+tests"
  # needs the UNTRACKED private terms file ($MPMF_HOME/private_terms.txt); a missing file fails.
  PYTHONDONTWRITEBYTECODE=1 MPMF_LIB_PATH="$src" "$PY" -m pytest "$src/../tests/test_name_gate.py" -q -p no:cacheprovider || { echo "DEPLOY REFUSED: name gate is red"; return 1; }
  echo "== full test suite: pytest tests (red fails the deploy)"
  # r92 step D: the whole suite, not just the script-style gates above. Run against the
  # candidate tree ($src/../tests), no bytecode/cache written into it. Explicit
  # `|| return 1` everywhere in this function: run_gate is called as `if ! run_gate`
  # in deploy mode, where bash ignores `set -e`, so a bare failing line would not stop it.
  PYTHONDONTWRITEBYTECODE=1 MPMF_LIB_PATH="$src" "$PY" -m pytest "$src/../tests" -q -p no:cacheprovider || { echo "DEPLOY REFUSED: pytest tests is red"; return 1; }
  cron_dryruns || return 1
}
cron_dryruns() {
  # r91a: every cron entry point must at least import and parse its flags against
  # a THROWAWAY MPMF_HOME. A deploy that ships a broken nightly/daily/watchdog/
  # capture/intake/historical command is refused here, not discovered at 04:30.
  local tmp py; tmp="$(mktemp -d /tmp/mpmf-cron-XXXXXX)"; py="$PY"
  mkdir -p "$tmp/eval" "$tmp/health"
  printf '{"store_path":"%s/dry.mpmf","store_mode":"attach"}' "$tmp" > "$tmp/recall.config.json"
  : > "$tmp/dry.mpmf"   # attach mode needs an existing file; the gate never writes to it
  local fail=0 name rc
  _run() { name="$1"; shift; if ( cd "$tmp" || exit 1; export MPMF_HOME="$tmp" MPMF_RECALL_CONFIG="$tmp/recall.config.json" MPMF_EVAL_DIR="$tmp/eval" MPMF_HEALTH_DIR="$tmp/health" MPMF_DRY_RUN=1 MPMF_LIB_PATH="$1" PYTHONPATH="$1"; shift; "$@" ) >/tmp/mpmf-cron-$name.log 2>&1; then rc=0; else rc=$?; fi; if [ $rc -eq 0 ]; then echo "cron dry-run OK   $name"; else echo "cron dry-run FAIL $name rc=$rc"; tail -3 /tmp/mpmf-cron-$name.log; fail=1; fi; }
  _run nightly      "$src" "$py" -m mpmf_plus.eval nightly --dry-run --split both --budget-usd 0.1
  _run daily        "$src" "$py" "$HOME/.hermes/scripts/mpmf_daily_checks.py" --dry-run
  _run watchdog     "$src" "$py" "$HOME/.hermes/scripts/mpmf_post_restart_watchdog.py" --dry-run
  _run capture      "$src" bash "$HOME/.hermes/scripts/thinking_capture_run.sh" --dry-run
  _run intake       "$src" "$py" -m mpmf_plus.ingest --dry-run
  _run historical   "$src" "$py" "$HOME/.hermes/scripts/historical_rumination.py" --dry-run
  _run re-enrich    "$src" "$py" "$HOME/.hermes/scripts/mpmf_re_enrich_pending.py" --dry-run
  # r92 step 1: the historical dry-run must reach the package intake gate and print
  # all six counts, and no live script may import athena_memory.
  if grep -qE "^intake_gate: staged=[0-9]+ merged=[0-9]+ promoted=[0-9]+ held=[0-9]+ rejected=[0-9]+ expired=[0-9]+" /tmp/mpmf-cron-historical.log; then
    echo "cron dry-run OK   historical gated path: $(grep -E '^intake_gate: staged=' /tmp/mpmf-cron-historical.log | head -1)"
  else
    echo "cron dry-run FAIL historical gated path: no intake_gate counts line"; fail=1
  fi
  rm -rf "$tmp"
  if [ $fail -ne 0 ]; then echo "DEPLOY REFUSED: a cron entry point is broken (see above)"; return 1; fi
  echo "== cron dry-runs: 7/7 OK"
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
# f1: the post-deploy watchdog judges only turns that STARTED after this instant
# (mpmf_plus.health.deploy_ts). Written only on success, i.e. after the gate and checkout.
HEALTH="${MPMF_HEALTH_DIR:-${MPMF_HOME:-$HOME/.mpmf}/health}"
mkdir -p "$HEALTH" && printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TAG" > "$HEALTH/deploy_at"
echo "DEPLOYED $TAG -> $(git -C "$LIVE" log --oneline -1)"
echo "Rollback: git -C $LIVE checkout --detach <previous tag> && /restart"
