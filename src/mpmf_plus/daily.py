"""MPMF daily checks — the real implementation (r98: moved out of ~/.hermes/scripts).

`python3 -m mpmf_plus.daily [--dry-run]`. The host script
(~/.hermes/scripts/mpmf_daily_checks.py) is now a thin shim that calls this module, so
the deploy gate's cron dry-run exercises the shipped code, not a copy on the host.

Silent when everything is fresh; prints one line per alert otherwise, plus the
thinking-capture counts line (r97, informational: it never sets the exit code).
"""
import sys


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    dry = "--dry-run" in argv or bool(__import__("os").environ.get("MPMF_DRY_RUN"))
    from .health import run_daily, daily_info, check_disk, check_known_stores
    from . import memory

    if dry:
        st = memory.store_settings()          # lazy: resolves only now
        print("DRY-RUN daily: store=%s mode=%s exists=%s"
              % (st.path, st.mode, __import__("os").path.isfile(st.path)))
        print("DRY-RUN daily: known stores=%d, checks=%s"
              % (len(check_known_stores()),
                 "store,thinking,scorecard,failures,known_stores,ingest_silence_turns,"
                 "capture_skips(info)"))
        print("DRY-RUN daily OK")
        return 0

    alerts = list(run_daily() or [])
    _d = check_disk(3.0)
    if _d:
        alerts.append(_d)
    for a in alerts:
        print(a)
    for line in daily_info():          # capture counts (captured / skipped by reason+model)
        print(line)
    return 1 if alerts else 0


if __name__ == "__main__":
    sys.exit(main())
