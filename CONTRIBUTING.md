# Contributing

Thanks for your interest in MPMF+.

## Contributor License Agreement (required)

**Every contribution requires a signed Contributor License Agreement (CLA).**
A pull request, patch, or email containing code, docs or tests is not accepted
until the CLA is agreed to in writing.

The CLA grants the author (feijaodo, the licensor) a perpetual, worldwide,
non-exclusive, royalty-free, irrevocable copyright licence to reproduce,
prepare derivative works of, publicly display, sublicense **and relicense**
your contribution under any licence — including licences other than the
PolyForm Noncommercial License 1.0.0 that governs this distribution. You keep
copyright in your contribution and may use it however you like elsewhere.

Why: this package is dual-track — free for noncommercial use, commercially
licensed by the author. Without the right to relicense, a contribution could
not be included in a commercial licence, so the CLA is a precondition, not a
formality.

## How to contribute

1. Open an issue describing the change (or claim an existing one).
2. Make the change small and self-contained; keep the "no silent fallbacks"
   contract: missing store / bad config / dead embedder must raise
   (`StoreNotFoundError`, `ConfigError`, `EmbedderError`).
3. Add or extend a test under `tests/`. Every claim must have an artifact.
4. Run `python -m pytest tests/` — green before you send.
5. State explicitly in the PR text: "I agree to the CLA in CONTRIBUTING.md."
   Unsigned contributions are closed, not merged.

## Reporting bugs

Include the store path shape (never the store itself), the exact exception
type and message, and the commit you ran. Do not attach real memory stores.
