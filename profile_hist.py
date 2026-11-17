"""R90e: profile the 5 slowest temporal prefetches (history ON). Read-only."""
import cProfile, io, json, pstats, sys, time
from collections import Counter

sys.path.insert(0, "/root/mpmf-release/src")
from mpmf_plus import provider as P
from mpmf_plus import history as H

P.configure(PATH="/root/.mpmf/athena.mpmf")
P._RECALL_CONFIG_PATH = "/root/mpmf-release/.r90e_history_on.json"
from mpmf_plus import hermes_adapter
prov = hermes_adapter.MPMFPlusProvider()
prov.initialize("r90e-profile")

rows = json.load(open("/root/r88/order_hist32.json")) + json.load(open("/root/r90d/order_on_rest.json"))
slow = sorted(rows, key=lambda r: -r.get("prefetch_ms", 0))[:5]
mm = P._open().mem

# index build cost
t = time.perf_counter(); idx = H.index().ensure(mm); build_ms = (time.perf_counter()-t)*1000
print("INDEX build_ms=%.1f nodes=%d entity_keys=%d" % (build_ms, len(idx.entity_of), len(idx.by_entity)))

orig_get = type(mm).get_text
calls = Counter(); gtime = [0.0]
def timed_get(self, nid):
    s = time.perf_counter()
    try: return orig_get(self, nid)
    finally: calls["get_text"] += 1; gtime[0] += (time.perf_counter()-s)*1000
type(mm).get_text = timed_get

orig_sel = H.select_relevant
sel = Counter()
def timed_sel(query, candidates, memory_manager, keep=12):
    t = time.perf_counter(); out = orig_sel(query, candidates, memory_manager, keep)
    sel["ms"] += (time.perf_counter()-t)*1000; sel["cands"] += len(list(candidates)) if not isinstance(candidates, list) else len(candidates); sel["n"] += 1
    return out
H.select_relevant = timed_sel

for r in slow:
    q = r["q"]
    before = dict(calls); gt0 = gtime[0]; s0 = dict(sel)
    pr = cProfile.Profile(); t = time.perf_counter(); pr.enable()
    ctx = prov.prefetch(q) or ""
    pr.disable(); ms = (time.perf_counter()-t)*1000
    buf = io.StringIO(); pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(40)
    lines = [l for l in buf.getvalue().splitlines() if any(k in l for k in
             ("get_text", "recall", "embed", "rerank", "select_relevant", "render_history", "ensure", "history"))]
    print("\nQ%s total=%.0f get_text=%d/%.0fms select=%d cands=%d/%.0fms block=%d hist=%s" % (
        r["id"], ms, calls["get_text"]-before.get("get_text", 0), gtime[0]-gt0,
        sel["n"]-s0.get("n", 0), sel["cands"]-s0.get("cands", 0), sel["ms"]-s0.get("ms", 0.0), len(ctx), "history ·" in ctx))
    for l in lines[:6]: print("  ", l.strip()[:150])
