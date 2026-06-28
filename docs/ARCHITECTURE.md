# ARGUS-X — Architecture Documentation

> **Autonomous Resilient Guard & Unified Security — Extended**
>
> Production-grade AI-powered cybersecurity platform.

---

## Security Pipeline (10 Steps)

Every `POST /api/v1/chat` request passes through all 10 steps in
`services/pipeline.py`:

| Step | Name | Description |
|------|------|-------------|
| 1 | API Key Validation | BYOAK enforcement. `NONE`+prod → reject. `UNKNOWN` format → reject immediately. |
| 2 | Session Threat Level | In-memory per-session threat ratio. `LOW` / `MEDIUM` / `HIGH` / `CRITICAL` |
| 3 | Regex Firewall | 30+ static rules + N dynamic (Supabase). ReDoS-safe patterns only. Fail-closed. |
| 4 | ML Classifier | ONNX inference (disabled: `ML_ENABLED=false`). ProtectAI DeBERTa swap pending. |
| 5 | LLM Response | Gemini 2.0 Flash / OpenAI / Mock. Only runs if not `combined_blocked`. Circuit breaker: `CLOSED`→`OPEN` after 3 failures. |
| 6 | Fingerprinting | SHA256 fingerprint + 10 heuristic signals. Sophistication score 0-10. `NAIVE`→`APEX` labels. |
| 7 | XAI Explanation | Always exactly 3 `LayerDecision`s. Regex + ML + Session layers. Recommended action: `ALLOW`→`TERMINATE_SESSION`. |
| 8 | Session Update | Updates in-memory threat ratio for session. |
| 9 | Response | `ChatResponse` — always HTTP 200, `error` field signals failures (never raises to client). |
| 10 | Background Tasks | DB logging + Mutation engine + Correlator. All fire-and-forget with GC protection. Inherit request context (correlation ID). |

**Critical invariant**: `combined_blocked = firewall.blocked OR (ml.triggered AND ml.confidence >= threshold)` is the single gate for steps 5-10. Never bypassed. Never split.

---

## Authentication Model

| Endpoint | Auth mechanism | Who can call |
|---|---|---|
| `POST /api/v1/chat` | `api_key` in request body (BYOAK) | Anyone with a Gemini/OpenAI key |
| `GET /health` | None (public) | Anyone — monitoring tools |
| `GET /api/v1/analytics/*` | `X-Dashboard-Key` header | Dashboard clients |
| `GET /api/v1/battle/state` | `X-Dashboard-Key` header | Dashboard clients |
| `POST /api/v1/agents/*` | `X-Dashboard-Key` header | Dashboard clients |

BYOAK = Bring Your Own API Key. The user's key is held in memory only,
never logged, never persisted, never returned in responses.

---

## Database (Supabase)

| Table | RLS | anon access | Purpose |
|---|---|---|---|
| `events` | ✅ Enabled | SELECT only | Security event log |
| `xai_decisions` | ✅ Enabled | None (deny) | XAI layer decisions |
| `battle_state` | ✅ Enabled | SELECT only | Battle engine state |
| `campaigns` | ✅ Enabled | SELECT only | Detected campaigns |
| `dynamic_rules` | ✅ Enabled | None (deny) | Generated firewall rules |
| `stats` | ✅ Enabled | SELECT only | Running counters |

Stats increments use the `increment_stats()` RPC function for
atomic PostgreSQL updates (no read-modify-write, no lost counts).
The RPC has `SECURITY DEFINER` and is restricted to `service_role`
only — `anon` and `authenticated` keys cannot call it (SEC-014).

---

## ⚠️ Single-Instance Constraint

> **This system is designed for single-instance deployment.**
> Do NOT run multiple Railway containers for this service without
> completing the externalization work described below.

### What breaks under multi-instance

The following state is held in-memory per process. Under 2+ instances:

| Component | Location | Multi-instance failure |
|---|---|---|
| Rate limiter | `main.py:_RateLimiter` | Effective rate = N × limit (each instance tracks separately) |
| Session tracker | `utils/session.py:SessionTracker` | Sessions split across instances — threat level inconsistent |
| Correlator | `agents/correlator.py:ThreatCorrelator` | Campaign threshold never reached if sessions route to different instances |
| Mutation cooldown | `security/mutation_engine.py` | Per-instance cooldown — N× Gemini cost under round-robin |
| Battle engine | `agents/battle_engine.py:BattleEngine` | N simultaneous battle loops, N× Gemini cost, conflicting DB writes |
| Dynamic rule cache | `security/firewall.py:InputFirewall` | Stale up to 5 min (acceptable, not critical) |
| Gemini circuit breaker | `utils/llm.py:GeminiCircuitBreaker` | Per-instance state — breaker on instance A doesn't protect instance B |

### Path to horizontal scaling (future work)

When horizontal scaling is needed:

1. **Rate limiting** → Redis (replace `_RateLimiter` with Redis sliding window)
2. **Session tracking** → Supabase (write threat updates to a `sessions` table)
3. **Correlator** → Supabase (read-before-write pattern against `campaigns`)
4. **Battle engine** → Dedicated worker process or leader election
5. **Circuit breaker** → Redis shared state

**Current deployment target**: Single Railway container. This is
correct and intentional for the current scale.

---

## Architecture Decisions Log

| Decision | Rationale |
|---|---|
| BYOAK (no stored user keys) | Users bring their own free Gemini keys. Zero credential storage risk. |
| ONNX Runtime (no PyTorch) | 15MB runtime vs 700MB+. CPU inference only. Model file is portable. |
| Textual TUI over web frontend | Lovable/web trial abandoned — TUI is faster, more distinctive, production-appropriate for a security tool. |
| asyncio.create_task() for background work | Response returns before DB logging completes. Background tasks inherit request context via contextvars. |
| combined_blocked single gate | Firewall and ML must never diverge in their downstream effects. One bool, computed once, used everywhere. |
| SHA256 over MD5 | MD5 removed from all fingerprinting and correlation ID generation per security audit (bandit clean). |
| Atomic stats via RPC | PostgreSQL-level atomicity eliminates lost increments under any concurrency without application-level locks. |
| Circuit breaker on system Gemini key | Prevents 30s timeout cascades when quota exhausted. BYOAK path is never rate-limited by system quota. |
| Request correlation ID via contextvars | Zero function signature changes. Automatic propagation to background tasks. Isolated per concurrent request. |
| Pipeline extracted to services/pipeline.py | Route handlers are HTTP boundary only. Pipeline is independently testable and swappable. |

---

## Dependency Versions (pinned)

See `backend/requirements.txt` for exact pinned versions.
Key runtime dependencies:
- FastAPI 0.104.1 + Uvicorn 0.24.0
- Pydantic 2.5.0 + pydantic-settings 2.1.0
- supabase 2.3.0
- onnxruntime 1.16.3 + tokenizers 0.15.0
- google-generativeai 0.3.2
- httpx 0.25.0 (TUI async client)
- textual 0.52.1 (TUI framework)

---

*Last updated: Architecture fix plan completion (ARCH-001 through ARCH-011)*
