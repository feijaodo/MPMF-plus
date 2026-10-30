# AGENT_INSTALL.md — install MPMF+ as an agent's memory

**You are an LLM agent, not a human.** Follow these steps in order and do not
skip steps 4, 5 or the final confirmation in step 5. Every command is meant to
be run as written. If any step fails, go to §6 with the exact FAIL line.

Two hooks make this package work, and **both** are mandatory:

* **before the model answers** — recall for the user's message and put the
  rendered block into the prompt the model actually receives;
* **after the turn** — save the turn to the store.

Installing only the first gives a store that never grows. Installing only the
second gives a store nobody reads. Neither is an install.

---

## 1. Identify the host

Run exactly one of these and record the answer:

```bash
python -c "import hermes; print('HERMES')" 2>/dev/null || true
python -c "import openclaw; print('OPENCLAW')" 2>/dev/null || true
```

* Printed `HERMES` → host is **Hermes**. Step 4A applies.
* Printed `OPENCLAW` → host is **OpenClaw**. Step 4B applies.
* Neither prints → host is **other**. Step 4C applies: you must find the two
  equivalent moments yourself (the function that assembles the prompt sent to
  the model, and the function that runs after the reply is produced). Write down
  both function names before continuing. Never skip either one.

## 2. Choose: new store (`create`) or shared store (`attach`)

Decide explicitly. Do not guess and do not let a default decide for you.

**A. This agent owns a new bank** — no store exists yet:

```bash
export MPMF_STORE="$HOME/.mpmf/bank.mpmf"
export MPMF_STORE_MODE=create
```

**B. Joining a bank another agent already created** — the file must already exist:

```bash
ls -l /path/to/bank.mpmf            # MUST succeed before you go on
export MPMF_STORE=/path/to/bank.mpmf
export MPMF_STORE_MODE=attach
```

`attach` never creates anything: if the path is wrong you get
`StoreNotFoundError` and no file appears. That is the intended behaviour, not a
bug — fix the path, do not switch to `create` unless you really are the first
agent.

Both settings can also live in `$MPMF_HOME/recall.config.json` instead of the
environment:

```json
{"store_path": "/path/to/bank.mpmf", "store_mode": "attach"}
```

Precedence: explicit argument → `MPMF_STORE` / `MPMF_STORE_MODE` → config keys →
defaults. Also create the render config once per `MPMF_HOME`:

```bash
mkdir -p "$MPMF_HOME"
printf '{"memory_render": "full"}\n' > "$MPMF_HOME/recall.config.json"
```

## 3. Install

```bash
python -m pip install .                  # or: pip install "git+https://github.com/feijaodo/mpmf-plus.git@v0.1.1"
python -m pip install ".[rerank]"        # optional cross-encoder (better order, ~1.2 GB with torch)
ollama pull mxbai-embed-large            # REQUIRED: the store is 1024-d mxbai
python -c "import mpmf_plus, mpmf; print(mpmf_plus.__file__)"
```

All agents sharing one store must use **this same embedder**. The store records
the vector width (1024) but **not which model produced the vectors**, so a
different model is silently wrong rather than detected. Check with:

```bash
curl -s http://127.0.0.1:11434/api/tags | grep -o mxbai-embed-large
```

## 4. Wire the two hooks

### 4A. Hermes

The package ships the binding; do not re-implement it.

```python
# ~/.hermes/plugins/mpmf_plus_bank/__init__.py
from mpmf_plus.hermes_adapter import register  # noqa: F401  (Hermes calls register(ctx))
```

Hermes then calls, in this order, on `MPMFPlusProvider`:

| moment | function | what it does |
|---|---|---|
| before the model answers | `MPMFPlusProvider.prefetch(query, session_id=...)` | `block.wrapper_recall()` → `block.render_memory_block()`; returns the block Hermes injects into the prompt |
| after the turn | `MPMFPlusProvider.sync_turn(user_content, assistant_content, session_id=...)` | `memory.store(...)` with `auto_extract=True`, `auto_link=False`, namespace `transcript` |

Verify both are actually called (not just registered):

```python
python - <<'PY'
from mpmf_plus.hermes_adapter import MPMFPlusProvider
p = MPMFPlusProvider({"top_k": 5}); p.initialize(session_id="install-check")
print("prefetch ->", len(p.prefetch("what do you remember?")), "chars")
p.sync_turn("a" * 80, "b" * 80, session_id="install-check")
print("sync_turn error ->", p.last_sync_error)   # must be None
PY
```

`sync_turn` logging `MPMF+ turn write FAILED` or a non-`None`
`last_sync_error` means hook 2 is broken — fix it before continuing.

### 4B. OpenClaw

Call the package directly at the two moments:

```python
# BEFORE the model answers — build the block and put it in the prompt
from mpmf_plus import block as _block
from mpmf_plus import memory as _memory

def memory_block_for_prompt(user_message: str, top_k: int = 8) -> str:
    shim = _block.wrapper_recall(user_message, top_k, _memory)
    if shim is None:
        return ""                                  # recall raised; surface it in logs
    body, _items = _block.render_memory_block(
        user_message, shim, _memory._open().mem, top_k, max_chars=3500)
    return body                                    # PREPEND this to the prompt

# AFTER the turn — persist it
def save_turn(user_message: str, assistant_reply: str, session_id: str) -> None:
    _memory.store(
        f"USER: {user_message[:600]}\nASSISTANT: {assistant_reply[:600]}",
        name=f"turn_{session_id}_{int(__import__('time').time() * 1000)}",
        auto_extract=True, auto_link=False, namespace="transcript")
```

Wire `memory_block_for_prompt` into the prompt-assembly function and `save_turn`
into the post-reply function. If your host only lets you hook one place, hook
prompt assembly and call `save_turn` at the end of the same turn.

### 4C. Other hosts

Find and name, in writing, these two functions before editing anything:

1. the one that assembles the message list / prompt string sent to the model —
   prepend the output of `memory_block_for_prompt` (4B) to the **user turn** (or
   to the system prompt, but never as a second user message);
2. the one that runs once the assistant reply is complete — call `save_turn`
   (4B) there.

Never skip either one. Never call `save_turn` from inside the prompt builder.

## 5. Self-test (mandatory — the install is not done until both parts pass)

**Part 1 — the mechanical gate:**

```bash
python -m mpmf_plus.selftest --store "$MPMF_STORE" --mode "$MPMF_STORE_MODE"
```

It writes a unique marker node, reads it back by name, recalls it by its own
text, renders the memory block and checks the marker is inside it. Exit code 0
and a final `PASS:` line are required. It prints the marker name — keep it.

**Part 2 — one real turn:** send one real user message through the host (the
full path: prompt assembly → model → reply) whose text contains the marker from
part 1, e.g.

> `memory check: reply with the marker you see — selftest-xxxxxxxx`

Then inspect **the prompt the model actually received** (host debug log, request
log, or the provider's own trace — not your guess) and confirm the marker string
appears in it. Confirm the reply echoes it. Only then is the install finished.

## 6. Troubleshooting — every FAIL the selftest can print

| FAIL line | meaning | fix |
|---|---|---|
| `1/6 resolve store` + `ConfigError: store_mode must be one of` | `$MPMF_STORE_MODE` or config `store_mode` is not `create`/`attach` | set it to exactly `create` or `attach` |
| `1/6 resolve store` + `ConfigError: ... empty` | blank mode/path was passed | remove the empty value; do not rely on a default you did not choose |
| `1/6 resolve store` + `ConfigError: recall config unreadable/not a JSON object` | `$MPMF_HOME/recall.config.json` is corrupt | rewrite it: `{"memory_render": "full"}` |
| `1/6 recall config` | config file missing | `mkdir -p "$MPMF_HOME" && printf '{"memory_render": "full"}\n' > "$MPMF_HOME/recall.config.json"` |
| `2/6 store_mode` + `StoreNotFoundError: attach: store not found` | you are attaching to a path that does not exist | fix the path; only the first agent uses `create` |
| `2/6 store_mode` + `FileExistsError: refusing to create` | `create` was pointed at an existing store | use `attach`, or choose a new path |
| `3/6 write marker` + `RuntimeError: local embed failed` | Ollama is not reachable/loaded | `ollama serve`; `ollama pull mxbai-embed-large`; check `OLLAMA_URL` |
| `3/6 write marker` + `DimensionMismatchError` | embedder is not 1024-d | use `mxbai-embed-large`, or a store built for that width |
| `3/6 write marker` + `LockTimeoutError` | another writer held the store lock too long | raise `MPMF_STORE_LOCK_TIMEOUT`; check no process is stuck |
| `3/6 write marker` + node count did not go +1 | dedup merged your marker (cosine ≥ 0.95) or a stale writer replaced the store | rerun on a scratch store; never test writes on a shared bank with near-duplicate text |
| `4/6 get() by name` | the write returned but the node is not readable | the write hit another file: check `store=` in step 1/6 equals what you set |
| `5/6 recall()` | node exists but retrieval misses it | embedder mismatch or empty store image; verify the embedder, then rerun |
| `6/6 render prompt block` | the renderer dropped the marker | render config missing/invalid; the block is what the model sees, so this is fatal |
| any `FAIL` in the host after a passing selftest | hooks are not both wired | recheck §4: `prefetch` output must be in the prompt; `sync_turn` must not log a failure |

## 7. Undo the install

```bash
# 1. Hermes: remove the provider plugin
rm -rf ~/.hermes/plugins/mpmf_plus_bank
#    OpenClaw / other: remove the two calls you added (§4B/4C) and the helper module

# 2. Remove the settings you exported / added
unset MPMF_STORE MPMF_STORE_MODE            # and delete store_path/store_mode keys
rm -f "$MPMF_HOME/recall.config.json"       # only if it was created for this install

# 3. Remove the package
python -m pip uninstall -y mpmf-plus

# 4. Leave the store file alone unless you are the last agent on it.
#    Yours to delete only if you created it and no other agent attaches to it:
#    rm -f "$MPMF_STORE" "$MPMF_STORE.lock"
```

Undo never touches a shared bank that another agent still uses: check
`MPMF_STORE` is your own file before deleting anything.
