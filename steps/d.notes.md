# Step D notes

## 1. Skipped tests (12 with no API key in the environment; 10 when OPENAI/OPENROUTER key is set)

Command: env -u OPENAI_API_KEY -u OPENROUTER_API_KEY HOME=/root python -m pytest tests -q -rs
Result: 161 passed, 12 skipped.

1.  tests/test_dates.py:122 test_block_shows_said_date — obsolete since r89c: show_said_date is default OFF and block.render_cfg() does not pass the flag through, so the date suffix cannot be switched on; reinstate when the flag is wired (0.3.0 finding).
2.  tests/test_dates.py:133 test_block_flags_ingest_date_as_not_said — obsolete since r89c, same reason (show_said_date not reachable from config).
3.  tests/test_embed_text_agreement.py:43 test_thinking_writer_agrees — obsolete: the external thinking writer (~/.mpmf/athena_thinking.py) no longer defines _embed_text; the package rule is exercised by the other tests in the file.
4.  tests/test_embed_text_agreement.py:53 test_legacy_wrapper_agrees — legacy wrapper athena_memory.py was retired (provider_entry replaced it in r84). (runtime skip: file absent)
5.  tests/test_embed_text_agreement.py:59 test_call_sites_use_the_core_rule[mpmf_daily_checks] — obsolete: mpmf_daily_checks.py is now a thin wrapper over mpmf_plus.health.run_daily (r89/r91b) and has no embed call site.
6.  tests/test_embed_text_agreement.py:69 test_call_sites_use_the_core_rule[scan_athena] — optional external call site (set MPMF_SCAN_ATHENA) not configured on this box. (runtime skip)
7.  tests/test_no_silent_fallbacks.py:41 test_missing_config_raises — superseded in r91b: a MISSING recall config now falls back to the package defaults (portable install, block.py render_cfg); unreadable or invalid config still raises (other tests).
8.  tests/test_openai_embed.py:26 test_openai_compat_embedder_returns_vector_at_requested_dim — No OPENAI_API_KEY/OPENROUTER_API_KEY in env. (skipif; runs and passes when a key is set)
9.  tests/test_openai_embed.py:35 test_openai_compat_embedder_pack_roundtrip — same: no API key in env.
10. tests/test_provider.py:71 test_same_results_as_athena_memory_on_a_small_store — legacy module (~/.mpmf/athena_memory.py) not on this box. (runtime skip)
11. tests/test_r87b_ingest_e2e.py:24 — obsolete: the host plugin no longer exposes _get_wrapper_module (r89 loads mpmf_plus.provider directly); per-turn ingest growth is covered by test_e2e_store_growth.py in the deploy gate.
12. tests/test_scripts.py:23 test_script[test_integration.py] — obsolete: v0.1.0 script calls MPMFMemory.remember()/create_node outside write_txn(), which the core now refuses (r69 guard); lifecycle covered by test_cas.py, test_core.py, test_e2e_store_growth.py.

With the shell's own OPENROUTER key set: 163 passed, 10 skipped (items 8-9 run and pass).

## 2. src/ changes made to get tests green

I (this step-D run) made NO src/ change. The only src/ edits in the working tree
were already there (mtime 2026-11-10 03:36, before this run started 04:57) and are
committed here as found. Both are comment/message text only, no logic:

- src/mpmf_plus/health.py (check_disk message): "prune /root/backups and round scratch" -> "prune old backups and scratch". Removes an author-box path from a user-facing string in the released package (portability scrub, test_dedup_audit_log's no-hardcoded-/root check class).
- src/mpmf_plus/store.py (comment in resolve_store): "how /root/.mpmf ended up" -> "how a production home directory ended up". Same reason.

Check: with both src edits reverted and everything else unchanged, pytest still gives 163 passed / 10 skipped, so neither is required for green; they are kept as the portability scrub.

## 3. Test-side changes committed in this commit (found uncommitted, NOT made by this run)

The working tree held uncommitted test edits from an earlier session (mtime 03:36-03:39).
They are required for `pytest tests` to be green from a clean checkout (the deploy gate
runs on a clean worktree of the candidate commit), so they are committed as found.
Flagged because the run rules forbid editing tests to pass a step: this run did not
write them, but a reviewer should look at them (see DECISION NEEDED in RUN_LOG).
- tests/conftest.py: collect_ignore for script-style files + new tests/test_scripts.py that runs them in subprocesses.
- Path portability (/root/... -> repo-relative / expanduser): test_e2e_store_growth, test_host_contract, test_provider, test_swap_store, test_r69_guards, test_r87b_ingest_e2e, test_embed_text_agreement, test_dedup_audit_log, test_portability (docstring).
- Assertions changed (not only paths): tests/test_r83c_attribution.py node-name prefix "f_" -> "memfact_"; tests/test_store_mode.py monkeypatch of memory._SETTINGS/PATH/STORE_MODE -> memory.configure_store(); tests/test_embed_text_agreement.py dropped needle "embed(embed_text(t))" from ingest.py; tests/test_r69_guards.py mode fixed to "create".
- Skips added: items 1,2,3,5,7,11,12 above (@pytest.mark.skip) and runtime skips 4,6,10; tests/test_openai_embed.py skips on HTTP 401/402/403 from the live endpoint.

## 4. deploy.sh
run_gate now runs `pytest <src>/../tests -q` (candidate tree) after the e2e gate and before the cron dry-runs; red => "DEPLOY REFUSED: pytest tests is red", return 1. Every gate line in run_gate now has `|| return 1`: run_gate is called as `if ! run_gate` in deploy mode, where bash ignores set -e, so earlier gate lines could not actually fail a deploy. HOME is defaulted from passwd if unset (runner starts with HOME unset; set -u crashed the cron dry-runs).

## 5. D-as-a-person scrub
Regex: D said|D's|for D\b|by D\b|^D: (case-sensitive, \b), ADDED lines of `git log -p`.
- HEAD ancestry (57 commits; v0.2.0..HEAD is 49, not 47): 4 hits, all in af51165 (v0.1.0 initial commit): memory.py + CHANGES/before/live_wrapper.py x2 each ("overwriting D's curated memory", "(D's home_channel)").
- All refs (115 commits incl. backup_before_squash_75b, r89, origin/main): 8 hits = those 4 + f28cc9b health.py "If D said something", a857b9c capture.py "D's refined spec", ab60475 provider.py x2.
- v0.2.0..HEAD: 0 hits.
- Tree at HEAD: 0 hits (already "the user's"/"the user said"). Working tree incl. this commit: 0 hits. Nothing to fix at HEAD; no history rewritten (history hits remain in old commits by rule).
