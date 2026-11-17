"""R90e: prefetch-only latency on all 64 order questions, same 4-thread shape as eval."""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/root/mpmf-release/src")
from mpmf_plus import provider as P
P.configure(PATH="/root/.mpmf/athena.mpmf")
P._RECALL_CONFIG_PATH = "/root/mpmf-release/.r90e_history_on.json"
from mpmf_plus import hermes_adapter
prov = hermes_adapter.MPMFPlusProvider()
prov.initialize("r90e-bench")
qs = json.load(open("/root/r88/questions_order.json"))

def one(x):
    t = time.perf_counter()
    ctx = prov.prefetch(x["q"]) or ""
    return (time.perf_counter() - t) * 1000, ("history ·" in ctx)

with ThreadPoolExecutor(int(os.environ.get("THREADS", "4"))) as ex:
    out = list(ex.map(one, qs))
ms = sorted(x[0] for x in out)
print(json.dumps({"n": len(ms), "p50": round(ms[len(ms)//2], 1), "p90": round(ms[int(.9*(len(ms)-1))], 1),
                  "max": round(ms[-1], 1), "hist": sum(1 for x in out if x[1])}))
