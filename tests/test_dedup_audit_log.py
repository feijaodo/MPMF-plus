"""r60c: the dedup audit log must follow $MPMF_HOME, never a hardcoded /root path.

Regression for a real leak: during the r60b stress run, probes written to a
scratch store in /tmp appended their dedup decisions to the author's LIVE
/root/.mpmf/health/dedup_audit.log, because the path was a module constant.
"""
import os

from mpmf_plus import dedup_helpers
from mpmf_plus.dedup_helpers import audit_log_path


def test_audit_log_follows_mpmf_home(monkeypatch, tmp_path):
    monkeypatch.setenv("MPMF_HOME", str(tmp_path))
    p = audit_log_path()
    assert p == tmp_path / "health" / "dedup_audit.log"
    assert not p.exists()  # resolving must not create anything


def test_audit_write_creates_health_dir_under_home(monkeypatch, tmp_path):
    monkeypatch.setenv("MPMF_HOME", str(tmp_path))
    assert not (tmp_path / "health").exists()
    dedup_helpers._audit_dedup("near_dup", kept_id=7, skipped_hash="abc",
                               namespace="memory", score=0.99)
    log = tmp_path / "health" / "dedup_audit.log"
    assert log.is_file(), "audit log was not created under $MPMF_HOME/health"
    body = log.read_text()
    assert "action=near_dup" in body and "kept_id=7" in body


def test_live_root_mpmf_untouched(monkeypatch, tmp_path):
    """Writing an audit line for a scratch home must not append to /root/.mpmf."""
    live = "/root/.mpmf/health/dedup_audit.log"
    before = None
    if os.path.isfile(live):
        st = os.stat(live)
        before = (st.st_size, st.st_mtime_ns)
    monkeypatch.setenv("MPMF_HOME", str(tmp_path))
    dedup_helpers._audit_dedup("exact_match", kept_id=1, skipped_hash="deadbeef")
    after = None
    if os.path.isfile(live):
        st = os.stat(live)
        after = (st.st_size, st.st_mtime_ns)
    assert before == after, f"{live} was modified by a scratch-home write"


def test_no_hardcoded_root_paths_in_src():
    """No runtime path in src/ may be pinned to /root or an absolute home."""
    src = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    offenders = []
    for root, _dirs, files in os.walk(src):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(root, fn)
            with open(p, encoding="utf-8") as f:
                for i, line in enumerate(f, 1):
                    s = line.strip()
                    if s.startswith("#") or s.startswith('"""') or s.startswith("*"):
                        continue  # provenance comments may name the author's box
                    if "/root/" in line and ("open(" in line or "=" in line):
                        offenders.append(f"{p}:{i}: {s}")
    assert not offenders, "hardcoded /root path(s) in runtime code:\n" + "\n".join(offenders)
