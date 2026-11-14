# DeepThink AIOS — Audit Findings, Improvements & Enhancement Roadmap

_Last updated: 2026-11-14. Produced from a deep scan of the backend (`backend/`, ~12.8k LOC)
and frontend (`frontend/`). Everything below is grounded in the actual code, not aspiration._

---

## 1. Bugs found & fixed this session

| # | Severity | Location | Bug | Fix |
|---|----------|----------|-----|-----|
| 1 | **High** | `backend/sandbox.py` | `Sandbox._strip_ansi` was called in 7 places but never defined → **every successful Python execution returned `ExecutionError`**. | Added the missing static method. |
| 2 | **High** | `backend/benchmarks/runner.py` | `stop_benchmark()` called `add_log()` while holding the non-reentrant `STATE_LOCK` → **`POST /api/cancel` deadlocked forever**. | Moved the log call outside the lock. |
| 3 | **High** | `backend/sandbox.py` | The "unrestricted fallback" (triggered whenever restricted mode hits an unlisted import) ran with **no isolation at all** — network, resource caps, PID/IPC namespaces were all bypassed. | Threaded `use_isolation=self.isolation_available` into both fallback runs, so the fallback is wrapped in `unshare --net`/RLIMIT containment. |
| 4 | **Medium** | `backend/vulkan_engine.py` | `load_installed_state()` / `save_installed_state()` referenced an **undefined `json`** (never imported); the `NameError` was swallowed by a broad `except`, so GPU-engine install state **silently never persisted** — load always returned defaults, save was a no-op. | Added `import json`. |

All fixes have regression tests (see §2). Full suite: **130 passing** under both `pytest`
and the dependency-free `backend/tests/run_tests.py`.

### Known, deliberate limitation (not a regression — needs a product decision)
The restricted sandbox's import whitelist is **not a hard security boundary**. When it blocks
an unlisted import, it re-runs the whole script in the fallback path. Fix #3 closed the
**network** and **resource-exhaustion** legs of that path, but the fallback still allows
**arbitrary filesystem read/write and `subprocess` execution** (there is no mount-namespace,
chroot, or seccomp-bpf filter). This is acceptable for a trusted single-user local tool, but
**not** for any multi-tenant or internet-facing deployment. See §3 for the three ways to close it.

---

## 2. Test coverage added

| File | Covers | Needs a model? |
|------|--------|----------------|
| `test_security.py` | SAST scanner, JWT (tamper/expiry/`alg=none`/forged-secret), air-gap config | No |
| `test_hardening.py` | SSRF in `scrape_url`, path traversal in `GitAgent` + `execute_workspace` | No |
| `test_sandbox_exec.py` | Real sandbox execution: output, errors, timeouts, bash, import fallback | No |
| `test_sandbox_fallback_isolation.py` | Fallback network containment (the fix #3 regression) | No |
| `test_api.py` | FastAPI `TestClient` over all non-LLM endpoints | No |
| `test_benchmark_state.py` | `stop_benchmark` deadlock + log-cap (fix #2 regression) | No |
| `test_router.py` | `TaskRouter.classify_task` deterministic keyword fast-path | No |
| `test_vulkan_state.py` | `vulkan_engine` install-state round-trip (fix #4 regression) | No |

CI: `.github/workflows/ci.yml` runs the backend suite + frontend build on every push
(lint is informational — see §4).

---

## 3. Improvements — prioritized

### P0 — Correctness & safety
1. **Decide the sandbox fallback policy.** Three options, with trade-offs:
   - **(a) Remove the fallback.** A blocked import becomes a hard, user-visible failure.
     Safest; breaks legitimate AI code that needs an unlisted-but-harmless package until the
     whitelist is widened.
   - **(b) Containerize the fallback** (Docker/gVisor/firejail/`bwrap` with a read-only rootfs
     and a tmpfs workdir). Keeps the UX, closes file + subprocess access. Highest correctness,
     adds a runtime dependency.
   - **(c) Widen the whitelist + add seccomp-bpf** so the fallback is rarely hit and, when it
     is, syscalls like `connect`/`execve`/`open` outside the workdir are denied in-process.
   - _Recommendation:_ (b) for any shared deployment; (c) as a lighter interim step.
2. **Stop swallowing errors silently.** Bug #4 hid for a long time because of
   `except Exception: pass`. Audit every bare/broad `except` around I/O and persistence
   (there are several in `vulkan_engine.py`, `memory.py`, `git_agent.py`); at minimum log them.
3. **Harden the SAST scanner** (`backend/security.py`). It only matches *literal* dangerous
   calls — `import os as o; o.system(...)`, `getattr(os,'sys'+'tem')`, and bare
   `import subprocess` all pass. Either treat it as advisory-only (rename so no one mistakes it
   for a boundary) or move enforcement to the OS layer (see P0.1).

### P1 — Robustness
4. **Concurrency:** `chat_lock` serializes all chat requests globally. Fine for single-user;
   document it, or move to per-session orchestrator state so two browser tabs don't block.
5. **JWT secret:** defaults to a per-PID random secret, so **all tokens are invalidated on
   every restart**. Fine for dev; for anything persistent, require `AIOS_JWT_SECRET` and fail
   loudly if auth is enabled without it.
6. **Config via env/pydantic-settings** instead of scattered `os.environ.get(...)` calls —
   one typed settings object, validated at startup.
7. **Structured logging** (`logging` module, levels, request IDs) instead of `print()`.

### P2 — Quality / DX
8. **Frontend lint debt:** `npm run lint` reports ~39 errors (unused imports like `React`,
   `Zap`, `Trash2`; `setState`-in-effect; empty blocks). None crash, but several unused state
   setters (`setContextLength`, `setMaxTokens`, `setTemperature`) suggest settings controls
   that may not be wired — worth a targeted look. Clean these, then flip CI lint to blocking.
9. **Frontend tests:** none exist. Add Vitest + React Testing Library for
   `utils/markdownParser.jsx` and `hooks/useChat.js` (pure-ish, high-value).
10. **Dead stores:** `mae`, `initial_failed_error`, `total_count`, unused `global`
    declarations — harmless, remove in a cleanup pass.

---

## 4. Extra features worth adding

- **Streaming correctness tests** for `/api/chat` (it's an SSE `StreamingResponse`) — assert
  event framing and that `generation_cancel` actually interrupts mid-stream.
- **End-to-end smoke test** with the pre-installed Playwright: boot the app with a tiny mock
  model, drive one prompt through each pipeline, screenshot. Catches wiring regressions the
  unit tests can't.
- **A real embedding backend** for `Memory` recall (the SQLite fallback is keyword-ish);
  wire an optional local sentence-transformer when ChromaDB is present.
- **Health/readiness endpoint** distinct from `/api/status` for container orchestration.
- **Request-level rate limiting** on the model-download and HF-search endpoints (they hit the
  network and can be abused).
- **A `make test` / `make lint` / `make dev`** or `justfile` so contributors have one entry point.

---

## 5. Sonnet 5.5 enhancement plan

The current orchestrator routes queries across **local SLMs/LLMs** (GGUF via Vulkan/llama.cpp).
Sonnet 5.5 (`claude-sonnet-5-5`) can be added as an **optional high-capability tier** without
breaking the "fully local" default. Phased, so each phase ships independently:

### Phase A — Pluggable model backend (foundation)
- Introduce a `ModelBackend` interface in `backend/orchestrator/` with two implementations:
  `LocalGGUFBackend` (today's path) and `AnthropicBackend` (Sonnet 5.5 via the Anthropic SDK).
- The orchestrator already has `_call_model(llm, prompt, max_tokens, temperature)` as the single
  choke point — route it through the backend interface so **no pipeline code changes**.
- Config: `AIOS_CLOUD_MODEL=claude-sonnet-5-5`, key via a secret (never an env var in plain
  text). Respect `air_gap` — when air-gap is on, the Anthropic backend is hard-disabled.

### Phase B — Smart routing (local ↔ cloud)
- Extend `TaskRouter` with a capability/difficulty signal: cheap/short/offline → local;
  hard reasoning, long context, or explicit user opt-in → Sonnet 5.5.
- Add a per-request `model_tier` field to `ChatRequest` ("local" | "cloud" | "auto") so the UI
  can expose the choice; default "local" to preserve current behavior.

### Phase C — Leverage Sonnet-5.5-specific strengths
- **Prompt caching** for the large static system prompts in the reasoning/chip-design pipelines
  (big cost/latency win on repeated calls).
- **Native tool use** for the coding pipeline: replace the hand-rolled "extract code → run in
  sandbox → feed error back" loop with structured tool calls (the sandbox becomes a tool the
  model invokes), which is more reliable than regex extraction.
- **Extended thinking** for the REASONING pipeline's proof/derivation tasks.
- **Vision** for the existing image path in `ChatRequest.image`.

### Phase D — Quality loop
- Add Sonnet 5.5 as an **LLM-judge** in `backend/benchmarks/evaluators.py` to grade local-model
  outputs against a rubric — turns the 11 benchmark suites into a continuous quality signal.
- Track cost/latency/accuracy per tier so routing (Phase B) can be tuned from data.

> **Before implementing any of Phase A–D:** load the `claude-api` skill for current model IDs,
> pricing, the tool-use / prompt-caching / extended-thinking APIs, and SDK usage — do not code
> the Anthropic integration from memory.

### Suggested first PR
Phase A only: the `ModelBackend` seam + `AnthropicBackend` behind a feature flag, with the
local path as the untouched default and a unit test using a mocked SDK client. Small, reviewable,
and unblocks everything else.
