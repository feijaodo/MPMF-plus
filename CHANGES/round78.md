# Round 78 — one copy of the core

* Live now loads `/root/mpmf-release/src` (`/root/mpmf` is a symlink), retiring
  the drifted `/root/mpmf` package of 2026-08-29 to `/root/mpmf.retired_<ts>`.
* The wrapper's dedup import now comes from `mpmf_plus.dedup_helpers` (package),
  not the vendored `/root/.mpmf/dedup_helpers.py` (kept as a fallback only).
* `check_dedup` near-dup pass: one float32 matvec + masks instead of a Python
  cosine per node. 540/540 verdicts identical; median 4,632 ms -> 270 ms.
* Callers fixed for the r69 guards (writes outside `write_txn` now raise):
  `athena_thinking.store_thinking` (node + edges), `intake_gate._rw_update`,
  `publish_lessons._update_existing`.
* Measured on a copy: store() p90 8,994 -> 3,692 ms, dedup 4,540 -> 226 ms,
  prefetch unchanged (381/440 -> 379/432 ms), top-8 parity 129/129.
