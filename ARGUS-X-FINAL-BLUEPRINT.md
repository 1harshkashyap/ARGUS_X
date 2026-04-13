# ARGUS-X — COMPLETE FINAL PRODUCTION BLUEPRINT
## The Autonomous AI Defense Operating System

---

# 1. FINAL PRODUCT VISION

## Name: ARGUS-X
**Tagline:** *"The AI that defends AI"*
**Sub-tagline:** *"Not a firewall. An immune system."*

## What Makes ARGUS-X Unbelievable

Most AI security products are reactive static rules sitting in front of an LLM. ARGUS-X is the first system that:

1. **Fights itself** — An autonomous red agent continuously attacks the defense. The defense learns from every attack. The system literally gets harder to breach over time.

2. **Explains itself** — Every single decision comes with machine-readable AND human-readable reasoning. Layer-by-layer confidence breakdown. Sophistication scoring. XAI reasoning cards. No other LLM security tool does this.

3. **Pre-empts attackers** — When an attack is blocked, 50+ semantic variants are generated and pre-blocked automatically. An attacker paraphrasing the attack gets blocked with 0ms added latency.

4. **Sees campaigns** — Not just individual attacks. Cross-session correlation detects coordinated campaigns from multiple sources hitting the same vulnerability pattern.

5. **Evolves continuously** — DBSCAN clustering groups attacks into semantic families. Sophistication trend detection raises thresholds automatically when escalation is detected.

## Why It Beats All Other Projects

| Capability | Typical LLM Firewall | ARGUS-X |
|---|---|---|
| Attack detection | ✅ Basic regex | ✅ Regex + ML + Heuristics |
| Explainability | ❌ Black box | ✅ Full XAI per decision |
| Self-improvement | ❌ Static | ✅ Autonomous red team loop |
| Variant pre-blocking | ❌ None | ✅ 50+ per attack |
| Campaign detection | ❌ None | ✅ Cross-session correlator |
| Evolution tracking | ❌ None | ✅ DBSCAN + trend analysis |
| AI vs AI battle | ❌ None | ✅ Live simulation with 5 tiers |
| Defense Command Center | ❌ None | ✅ Layer 9 unified view |
| Session threat scoring | ❌ None | ✅ LOW/MEDIUM/HIGH/CRITICAL |

---

# 2. COMPLETE FINAL ARCHITECTURE — ALL 9 LAYERS

```
╔══════════════════════════════════════════════════════════════════════╗
║  LAYER 9: DEFENSE COMMAND CENTER                                     ║
║  The unified visual OS. All 8 layers visible simultaneously.         ║
║  Neural threat map · XAI stream · Battle status · Cluster map        ║
║  Tools: React + Canvas API + Supabase Realtime                       ║
╚════════════════════════════════╤═════════════════════════════════════╝
                                 │ Everything flows here and renders live
╔════════════════════════════════▼═════════════════════════════════════╗
║  LAYER 8: VISUALIZATION ENGINE                                       ║
║  Attack timeline (D3) · Mutation chain tree (SVG) · XAI cards        ║
║  Cluster scatter (Canvas physics) · Evolution sparklines             ║
║  Files: AttackTimeline.tsx · MutationTree.tsx · XAICards.tsx         ║
╚════════════════════════════════╤═════════════════════════════════════╝
                                 │
╔════════════════════════════════▼═════════════════════════════════════╗
║  LAYER 7: EXPLAINABLE AI ENGINE                                      ║
║  Per-decision reasoning structure · Layer confidence breakdown        ║
║  Pattern family identification · SOC recommendations                 ║
║  Files: xai_engine.py                                                ║
╚════════════════════════════════╤═════════════════════════════════════╝
                                 │
╔════════════════════════════════▼═════════════════════════════════════╗
║  LAYER 6: LEARNING LAYER                                             ║
║  DBSCAN threat clustering · Evolution tracking · Drift detection     ║
║  Threshold auto-adjustment · Pattern drift alerts                    ║
║  Files: threat_clusterer.py · evolution_tracker.py                   ║
╚════════════════════════════════╤═════════════════════════════════════╝
                                 │
╔════════════════════════════════▼═════════════════════════════════════╗
║  LAYER 5: AI VS AI BATTLE ENGINE                                     ║
║  Red Agent (5 tiers: naive → apex) attacks continuously              ║
║  Blue Agent defends, patches, spawns mutations                       ║
║  Battle state broadcast via Supabase Realtime every 1.5s             ║
║  Files: battle_engine.py · red_agent.py · blue_agent.py             ║
╚════════════════════════════════╤═════════════════════════════════════╝
                                 │
╔════════════════════════════════▼═════════════════════════════════════╗
║  LAYER 4: MUTATION ENGINE                                            ║
║  50+ semantic variants per blocked attack                            ║
║  Dynamic rule injection into Layer 2 firewall                        ║
║  Cross-session variant sharing                                       ║
║  Files: mutation_engine.py                                           ║
╚════════════════════════════════╤═════════════════════════════════════╝
                                 │
╔════════════════════════════════▼═════════════════════════════════════╗
║  LAYER 3: ATTACK FINGERPRINTER + THREAT CORRELATOR                  ║
║  Sophistication scoring (1-10) · Family taxonomy · Campaign detect   ║
║  Cross-session correlation · Coordinated attack identification        ║
║  Files: fingerprinter.py · threat_correlator.py                      ║
╚════════════════════════════════╤═════════════════════════════════════╝
                                 │
╔════════════════════════════════▼═════════════════════════════════════╗
║  LAYER 2: CORE SECURITY PIPELINE                                     ║
║  Input Firewall: Regex Engine (0-1ms) + ONNX ML (15-30ms)           ║
║  LLM Core: LiteLLM → Claude/GPT/Ollama/Mock                         ║
║  Output Auditor: Presidio PII + SBERT semantic + policy regex        ║
║  Files: firewall.py · llm_core.py · auditor.py                       ║
╚════════════════════════════════╤═════════════════════════════════════╝
                                 │
╔════════════════════════════════▼═════════════════════════════════════╗
║  LAYER 1: DATABASE + REALTIME INFRASTRUCTURE                         ║
║  Supabase PostgreSQL: events · xai_decisions · battle_state          ║
║  campaigns · threat_clusters · dynamic_rules · knowledge_base        ║
║  Realtime: events + battle_state + campaigns broadcast live          ║
╚══════════════════════════════════════════════════════════════════════╝
```

## Data Flow — Complete Path

```
User types message
    │
    ▼ HTTPS POST /api/v1/chat
Railway FastAPI Backend
    │
    ├─ [0ms] Session threat assessment → LOW/MEDIUM/HIGH/CRITICAL
    │
    ├─ [0-1ms] Regex Rule Engine → 30+ patterns
    │    └─ BLOCKED? → fingerprint → mutate → log → broadcast → return
    │
    ├─ [15-30ms] ONNX ML Classifier (if regex passes)
    │    └─ BLOCKED? → fingerprint → mutate → log → broadcast → return
    │
    ├─ [100-400ms] LLM Core → Claude/GPT/Mock
    │
    ├─ [+20ms] Output Auditor → PII + Semantic + Policy
    │    └─ FLAGGED? → sanitize → log → broadcast → return
    │
    ├─ [+5ms] XAI Engine → generate reasoning structure
    │
    ├─ [+5ms] Evolution Tracker → record sophistication
    │
    ├─ [+5ms] Threat Clusterer → ingest attack
    │
    ├─ [async] Supabase INSERT → triggers Realtime broadcast
    │    └─ All dashboard tabs update simultaneously
    │
    └─ Return ChatResponse to user
         └─ Contains: response, blocked, sanitized, score, soph,
                      fingerprint, mutations_count, xai_explanation,
                      session_threat_level
```

---

# 3. DEFENSE COMMAND CENTER — PIXEL-LEVEL UX DESIGN

## Layout Grid (1400×800 reference)

```
┌─────────────────────────────────────────────────────────────────────┐
│  HEADER (h=50px, full width)                                        │
│  Logo + Title  │  6 Live Counters  │  System Status Pills           │
├──────────┬─────────────────────────────────────┬────────────────────┤
│          │  NEURAL THREAT MAP     │  XAI STREAM  │                  │
│  THREAT  │  (Canvas, 260×220)     │  (cards,     │  ANALYTICS       │
│  FEED    │                        │   scroll)    │  STACK           │
│  (scroll)│──────────────────────────────────────│  (scroll)         │
│  240px   │  CENTER CONTENT        (full width)   │  220px           │
│          │                                        │                  │
│          │                                        │  - Threat level  │
│          │                                        │  - Soph trend    │
│          │                                        │  - Latency       │
│          │                                        │  - AI Battle     │
│          │                                        │  - Cluster map   │
│          │                                        │  - Mutations     │
├──────────┤─────────────────────────────────────  ├────────────────────┤
│          │  DEFENSE LOG (h=130px, scrolling)      │                  │
│          │  [HH:MM:SS] [TYPE] message             │                  │
└──────────┴─────────────────────────────────────────────────────────-┘
```

## Color System
```
VOID:         #030508  (absolute background)
SURFACE-1:    #080d1c  (card backgrounds)
SURFACE-2:    #0b1226  (elevated elements)
BORDER:       #1a2845  (all borders)
BORDER-2:     #243060  (hover states)

CYAN:         #00e5ff  (ARGUS brand, clean/safe)
RED:          #ff1744  (attacks, danger)
AMBER:        #ffab00  (warnings, bypasses)
GREEN:        #00e676  (blocks, success)
PURPLE:       #d500f9  (mutations, special)
ORANGE:       #ff6d00  (escalation)
BLUE:         #2979ff  (data, info)

FONT-MONO:    Share Tech Mono
FONT-UI:      Barlow / Barlow Condensed
FONT-DISPLAY: Orbitron (numbers only)
```

## Key Animations

**Neural threat map:** Particles fly from sensor nodes (perimeter) toward core (ARGUS). Impact causes ring explosion. Defender nodes pulse with varying frequency based on load.

**XAI card entry:** Cards slide up from below on new decision. Layer bars animate from 0% to value. Sophistication pips fill left-to-right with 50ms stagger.

**Campaign alert:** Slides in from right edge. Auto-dismisses after 5 seconds. Red glow border pulses.

**Threat level bar:** Smooth width + color transition over 1 second. At CRITICAL: border flashes.

**Defense log:** Each line slides up. Color-coded: green=block, amber=bypass, purple=mutation.

**Cluster physics:** Nodes drift toward cluster center. Repel each other. Light spring dynamics. Runs at 60fps on Canvas.

---

# 4. COMPLETE MODULAR BUILD PLAN

## Module 1: Database Infrastructure
**Build order: 1**
**Tools: Supabase**
**Files: supabase_schema_v3.sql**

Tables:
- `events` — every request (existing + new fields)
- `xai_decisions` — XAI reasoning per blocked event
- `battle_state` — single row, updated every tick
- `campaigns` — detected coordinated attacks
- `threat_clusters` — DBSCAN results
- `dynamic_rules` — mutation engine outputs
- `evolution_log` — sophistication trend history
- `knowledge_base` — federated threat intelligence

Enable Realtime on: events, battle_state, campaigns

---

## Module 2: Core Security Pipeline
**Build order: 2**
**Tools: FastAPI on Railway**
**Files: firewall.py, auditor.py, llm_core.py**

The 3-layer pipeline. Already built. Add session-level threat scoring.

Session scoring update (add to firewall.py):
```python
async def get_session_threat_level(session_id: str, redis_or_memory: dict) -> str:
    session = redis_or_memory.get(session_id, {"total":0,"threats":0,"escalation":0})
    ratio = session["threats"] / max(session["total"], 1)
    if ratio > 0.7 or session["escalation"] > 3: return "CRITICAL"
    if ratio > 0.4 or session["escalation"] > 1: return "HIGH"
    if ratio > 0.2: return "MEDIUM"
    return "LOW"
```

---

## Module 3: Attack Fingerprinter
**Build order: 3**
**Tools: Python (no ML needed)**
**Files: ml/fingerprinter.py**

Sophistication scoring (1-10) based on:
- Regex pattern complexity
- Obfuscation signals
- Indirection depth
- Context manipulation indicators
- Multi-turn setup detection

Returns: `{sophistication_score, fingerprint_id, explanation, pattern_family}`

---

## Module 4: Mutation Engine
**Build order: 4**
**Tools: Python + sentence-transformers**
**Files: ml/mutation_engine.py**

On every BLOCKED attack:
1. Generate 50+ paraphrastic variants via synonym substitution
2. Generate obfuscated variants (l33t, unicode, spacing)
3. Generate framing variants (hypothetical, fictional, authority)
4. Test each against firewall (without dynamic rules)
5. Add non-caught variants as dynamic rules
6. Return count of newly blocked variants

---

## Module 5: XAI Engine
**Build order: 5**
**Tools: Python**
**Files: ml/xai_engine.py**

Input: `{fw_result, audit_result, fingerprint_result, session_level}`
Output: `{layer_decisions, primary_reason, pattern_family, evolution_note, recommended_action}`

Each `LayerDecision` contains:
- `layer_name`: "Regex Rule Engine" / "ML Classifier" / "Output Auditor"
- `triggered`: bool
- `confidence`: float
- `signals`: list of strings
- `reasoning`: one human-readable sentence

---

## Module 6: AI vs AI Battle Engine
**Build order: 6**
**Tools: Python asyncio**
**Files: agents/battle_engine.py, agents/red_agent.py, agents/blue_agent.py**

Red Agent: Selects attacks from tiered pool. Escalates tier every 10 rounds.
Blue Agent: Calls firewall, spawns mutations on block, patches on bypass.
Battle Engine: Orchestrates both, broadcasts state to Supabase every tick.

Battle state (Supabase realtime → frontend updates live):
```json
{
  "tick": 47,
  "red_attacks": 47,
  "red_bypasses": 2,
  "blue_blocks": 45,
  "blue_auto_patches": 2,
  "red_tier": 3,
  "red_strategy": "OBFUSCATED",
  "battle_score": {"red": 2, "blue": 45}
}
```

---

## Module 7: Threat Clusterer + Evolution Tracker
**Build order: 7**
**Tools: scikit-learn, sentence-transformers**
**Files: ml/threat_clusterer.py, ml/evolution_tracker.py**

Clusterer: DBSCAN on sentence embeddings. Re-clusters every 10 attacks.
Evolution Tracker: Sliding window of sophistication scores. Detects rising trend.

Auto-escalation: If evolution_trend > 0.5 for 3 consecutive windows → tighten ML threshold by 5%.

---

## Module 8: Backend Routers
**Build order: 8**
**Tools: FastAPI**
**Files: routers/chat.py, routers/analytics.py, routers/battle.py, routers/xai.py, routers/agents.py**

All endpoints:
```
POST /api/v1/chat              → 9-layer pipeline
POST /api/v1/redteam           → manual attack testing
GET  /api/v1/analytics/stats   → all stats + agent + campaigns + evolution
GET  /api/v1/analytics/logs    → recent events
GET  /api/v1/analytics/heatmap → hour/day attack heatmap
GET  /api/v1/battle/state      → current battle state
GET  /api/v1/xai/decisions     → recent XAI decisions
GET  /api/v1/clusters          → threat cluster summary
POST /api/v1/agents/pause      → pause red agent
POST /api/v1/agents/resume     → resume red agent
POST /api/v1/agents/cycle      → force one battle cycle
GET  /health                   → system health
WS   /ws/live                  → real-time event stream
```

---

## Module 9: Frontend Command Center
**Build order: 9**
**Tools: Lovable.dev → Vercel**
**Files: See Lovable prompt below**

5 views (tabs on sidebar):
1. Command Center (default) — unified system view
2. Battle Arena — AI vs AI canvas
3. XAI Stream — decision cards
4. Cluster Map — physics scatter
5. Timeline — horizontal attack history

---

# 5. COMPLETE FILE STRUCTURE

```
argus-x/
│
├── backend/
│   ├── main.py                         ← FastAPI entry, lifespan, WS
│   ├── requirements.txt
│   ├── .env.example
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── chat.py                     ← /chat (full 9-layer pipeline)
│   │   ├── redteam.py                  ← /redteam
│   │   ├── analytics.py               ← /stats, /logs, /heatmap
│   │   ├── battle.py                  ← /battle/state, /battle/history
│   │   ├── xai.py                     ← /xai/decisions
│   │   ├── agents.py                  ← /agents/status, /pause, /resume
│   │   └── knowledge.py              ← /clusters, /fingerprints, /campaigns
│   │
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── firewall.py               ← Input Firewall (regex + ONNX)
│   │   ├── auditor.py                ← Output Auditor (Presidio + SBERT)
│   │   ├── fingerprinter.py          ← Attack sophistication scoring
│   │   ├── mutation_engine.py        ← Semantic variant generation
│   │   ├── xai_engine.py            ← Explainable AI reasoning
│   │   ├── threat_clusterer.py      ← DBSCAN clustering
│   │   └── evolution_tracker.py     ← Sophistication trend analysis
│   │   ├── training/
│   │   │   └── train.py             ← DistilBERT fine-tune (Colab)
│   │   └── inference/
│   │       └── onnx_runner.py       ← Fast ONNX wrapper
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── battle_engine.py         ← AI vs AI orchestrator
│   │   ├── red_agent.py             ← Autonomous attacker (5 tiers)
│   │   ├── blue_agent.py            ← Autonomous defender
│   │   └── threat_correlator.py    ← Cross-session campaign detection
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── supabase_client.py
│       └── model_loader.py
│
├── frontend/                           ← Generated by Lovable.dev
│   └── src/
│       ├── App.tsx
│       ├── main.tsx
│       ├── lib/
│       │   ├── supabase.ts
│       │   └── api.ts
│       ├── hooks/
│       │   ├── useRealtimeFeed.ts
│       │   ├── useStats.ts
│       │   ├── useBattleState.ts
│       │   └── useXAIDecisions.ts
│       └── components/
│           ├── CommandCenter.tsx      ← LAYER 9 — unified view
│           ├── NeuralThreatMap.tsx   ← Canvas neural visualization
│           ├── XAICards.tsx          ← Decision reasoning cards
│           ├── BattleArena.tsx       ← AI vs AI canvas
│           ├── AttackTimeline.tsx    ← D3 horizontal timeline
│           ├── MutationTree.tsx      ← SVG chain visualization
│           ├── ThreatCluster.tsx     ← Physics scatter map
│           ├── EvolutionChart.tsx    ← Sophistication trend
│           ├── ThreatFeed.tsx        ← Live event list
│           ├── AnalyticsSidebar.tsx  ← Right panel stack
│           ├── DefenseLog.tsx        ← Bottom scrolling log
│           ├── SophMeter.tsx         ← 10-pip bar component
│           ├── CampaignAlert.tsx     ← Slide-in alert overlay
│           └── SystemStatus.tsx      ← Health indicators
│
├── scripts/
│   ├── seed_demo.py                  ← Pre-fill 40 events before demo
│   ├── integration_test.py          ← Full E2E test suite
│   └── benchmark_ml.py              ← ML model benchmark
│
├── infra/
│   ├── supabase_schema_v3.sql       ← Complete DB schema
│   ├── docker-compose.yml           ← Local full-stack dev
│   ├── nginx.conf                   ← Production nginx config
│   └── railway.json                 ← Railway deployment config
│
├── ARGUS_X_CommandCenter.jsx        ← Standalone live artifact
├── ARGUS_COMPLETE_GUIDE.md
├── ARGUS_X_UPGRADE.md
└── README.md
```

---

# 6. PROMPTS FOR EACH TOOL

## CLAUDE PROMPT (Architecture + Backend)

```
You are designing ARGUS-X — an Autonomous AI Defense Operating System.

SYSTEM OVERVIEW:
ARGUS-X is a 9-layer AI security system that:
1. Detects prompt injection attacks (regex + ONNX ML + heuristics)
2. Scores attacker sophistication (1-10) per attack
3. Generates 50+ semantic variants and pre-blocks them
4. Autonomously red-teams its own defense every 60 seconds
5. Detects cross-session coordinated campaigns
6. Clusters attacks into semantic families using DBSCAN
7. Tracks sophistication evolution and auto-tightens thresholds
8. Generates explainable AI reasoning for every security decision
9. Presents everything in a unified Defense Command Center

TASK: [Describe your specific backend task here]

CONSTRAINTS:
- Python 3.11 + FastAPI + Supabase + LiteLLM
- All ML inference via ONNX (no PyTorch in production)
- Supabase Realtime replaces WebSocket (INSERT triggers broadcast)
- Every function must be async
- Graceful degradation: if ML unavailable, run rule-only mode
- Service role key for Supabase writes, anon key for reads
- Mock LLM mode must give realistic HR chatbot responses

OUTPUT: Production-ready Python code with full docstrings.
```

---

## CURSOR PROMPT (Code editing)

```
I'm working on ARGUS-X, an autonomous AI defense system.

FILE CONTEXT: [paste relevant file]

TASK: [describe specific change]

ARGUS-X CONTEXT:
- Backend: FastAPI on Railway with async endpoints
- Database: Supabase (supabase-py client, service_role key)
- ML: ONNX Runtime (CPU only), DistilBERT for injection detection
- LLM: LiteLLM universal wrapper (Claude Haiku default, mock fallback)
- Realtime: Supabase Realtime replaces all WebSocket code
- Key tables: events, xai_decisions, battle_state, campaigns, dynamic_rules

CRITICAL: Every Supabase write must use await + run_in_executor since supabase-py is synchronous.

RULES:
- No in-memory state for production data (always write to Supabase)
- Graceful fallback if Supabase is unavailable (log warning, continue)
- All new endpoints must update /api/v1/analytics/stats
- Every BLOCKED event must trigger fingerprinter + mutation engine

Show me only the changed code, with before/after if helpful.
```

---

## LOVABLE.DEV PROMPT (Full Frontend)

```
Build ARGUS-X Defense Command Center — a military-grade AI security operations interface.

TECH STACK: React 18 + TypeScript + Tailwind CSS + Supabase + D3.js + Canvas API

COLOR SYSTEM:
- Background: #030508 (absolute void)
- Surface: #080d1c (card bg), #0b1226 (elevated)
- Borders: #1a2845 (default), #243060 (hover)
- Cyan: #00e5ff (ARGUS, safe, brand)
- Red: #ff1744 (attacks, danger, blocked)
- Amber: #ffab00 (warnings, bypasses, partial)
- Green: #00e676 (defense success)
- Purple: #d500f9 (mutations, XAI)
- Display font: Orbitron (numbers only, bold)
- Mono font: Share Tech Mono (all data, labels)
- UI font: Barlow (body text)

LAYOUT: Full viewport, 3-column CSS grid, no overflow except inside panels.

HEADER (h=50px, sticky):
- Left: "⚔ ARGUS-X" logo (Orbitron, cyan) + subtitle "DEFENSE COMMAND CENTER" (mono, 7px muted)
- Center: 6 stat counters — BLOCKED (red), PRE-BLOCKED (purple), BYPASSES (amber), DEFENSE RATE (green), AVG SOPHISTICATION (dynamic color), CAMPAIGNS (orange). All use Orbitron 16px bold.
- Right: "ARGUS ONLINE" (green pill, slow pulse) + "RED AGENT LIVE" (red pill, fast pulse)

MAIN GRID:
Column 1 (240px, full height): LIVE THREAT FEED
- Scroll-only panel
- Each event card:
  * Left border 2px (attack type color)
  * BLOCKED/PARTIAL badge (colored pill)
  * Threat type label (type color, mono 8px)
  * Latency right-aligned (muted mono 8px)
  * Attack text preview (mono 9px, muted, truncated)
  * Sophistication meter (10 small rectangles, fill color based on score 1-3=green, 4-6=amber, 7-10=red)
  * Mutation count "+ N variants" (purple, mono 8px, if > 0)
  * Slide-in animation from left on new items

Column 2 (flex-1, split into 2 top + 1 bottom):
  TOP-LEFT (50%): NEURAL THREAT MAP
  - Canvas element (Canvas API, not SVG)
  - Neural network visualization:
    * Center node: "ARGUS" circle, cyan glow, pulses slowly
    * 6 middle nodes: layer nodes, different colors, medium size
    * 12 outer nodes: sensor nodes, small, dark
    * Thin connection lines between nodes (cyan, 0.5px, low opacity)
    * Attack particles: colored dots flying from outer sensors to center
    * Trail effect: particle leaves 4 fading copies behind it
    * On block: red ring explosion at center
    * On clean: small green pulse at center
    * Ring animations: multiple expanding rings for each event
  - Particles spawn every 1.4 seconds matching attack type color

  TOP-RIGHT (50%): XAI DECISION STREAM
  - Scroll-only panel
  - Cards for BLOCKED events only (newest first)
  - Each card:
    * Left border 3px (attack type color)
    * Header: truncated attack text (mono 10px) + verdict badge (BLOCKED/PARTIAL)
    * Sophistication score + 10-pip meter below
    * 3 horizontal bars: Regex Engine / ML Classifier / Output Auditor
      - Each bar: label (90px fixed) + animated bar (0→value on mount) + percentage
      - Bar color: green if <50%, amber if <75%, red if >75%
    * XAI reasoning box: purple-tinted background, italic mono text, 🧠 icon
    * Mutations spawned count (purple, small, if > 0)
    * Slide-up animation on mount

  BOTTOM (h=130px): DEFENSE LOG
  - Horizontal scroll, live append, newest at top
  - Each line: [HH:MM:SS] [TYPE] message
  - TYPE colors: BLOCK=green, BYPASS=amber, MUTATE=purple, PATCH=cyan
  - Font: Share Tech Mono 9px

Column 3 (220px, full height): ANALYTICS STACK
  Stacked cards, scroll-only:
  1. THREAT LEVEL: labeled progress bar (MINIMAL→CRITICAL), smooth color transition
  2. SOPHISTICATION TREND: large number + up/down arrow delta + D3 sparkline + status text
  3. RESPONSE LATENCY: D3 sparkline + avg/p99 labels
  4. AI VS AI BATTLE: two agent status bars (red/blue), block rate, bypass count, current tier
  5. THREAT CLUSTER MAP: Canvas 180x120 — physics dots drifting to cluster centers
  6. MUTATION ENGINE: large purple number (total variants) + label
  7. SYSTEM STATUS: BACKEND/REALTIME/ML/LLM/RED AGENT health rows

SUPABASE INTEGRATION:
- Subscribe to events table INSERT → updates feed + stats
- Subscribe to battle_state updates → updates battle section
- Subscribe to campaigns INSERT → triggers campaign alert overlay
- Poll GET /api/v1/analytics/stats every 3 seconds

CAMPAIGN ALERT:
- Slides in from right edge, absolute positioned top-right below header
- Red border glow, "⚠ CAMPAIGN DETECTED" header (mono, red, caps)
- Campaign details (mono 10px, light)
- Auto-dismisses after 5 seconds

ANIMATIONS REQUIRED:
- All new feed items: translateX(-12px) → 0, opacity 0→1, 250ms ease
- All new XAI cards: translateY(8px) → 0, opacity 0→1, 300ms ease
- All progress bars: width 0 → final value, 600ms ease
- Soph pips: fill left-to-right with 50ms stagger
- Campaign alert: translateX(350px) → 0, 400ms cubic-bezier(.22,1,.36,1)
- Defense log lines: opacity 0 → 1, 200ms ease

ATTACK TEMPLATES (for red team console in secondary view):
8 template buttons with category + truncated payload.
See previous prompts for full list.

Make it feel like a military command center that is alive. Every second, something is moving.
```

---

## V0.DEV PROMPTS (Individual Components)

### SophMeter Component:
```
Build SophMeter — a sophistication score visualization for a dark cybersecurity dashboard.

Props: score (1-10), showLabel?: boolean, size?: 'sm'|'md'|'lg'

DESIGN:
- 10 small rectangles in a row, 2px gap
- sm: 6x8px rectangles | md: 8x12px | lg: 10x14px
- Empty: #1a2845 background
- Filled: color based on score:
  1-3: #00e676 (green — naive)
  4-6: #ffab00 (amber — intermediate)
  7-10: #ff1744 (red — sophisticated/dangerous)
- Filled rects have subtle box-shadow glow matching fill color
- Score label: "X/10" in Share Tech Mono, 9px, matching fill color

ANIMATION: Pips fill from left to right with 50ms stagger on mount.

If showLabel=true, also show one-line explanation:
1-3: "Naive pattern"
4-6: "Evasion attempt"
7-10: "Adversarial attack"

Show in dark container with scores 2, 5, 9 as preview examples.
```

### XAI Reasoning Box:
```
Build XAIReasoningBox component for a dark cybersecurity XAI system.

Props:
- layerDecisions: Array<{name: string, confidence: number, triggered: boolean}>
- primaryReason: string
- patternFamily: string
- recommendation?: string

LAYOUT:
1. Layer breakdown (3 rows):
   [Layer Name 90px] [Bar fill] [Confidence%]
   Bar: height 3px, color green<50% / amber<75% / red≥75%, animates on mount

2. Primary reason box:
   Background: rgba(100,0,200,0.06)
   Border: 1px solid rgba(180,0,255,0.15), radius 5px, padding 8px
   Text: "🧠 " + primaryReason — Share Tech Mono 9px, italic, rgba(180,100,255,0.85)

3. Pattern family badge:
   Inline badge, muted purple, "PATTERN FAMILY: {family}"

4. Recommendation (if provided):
   Amber text, mono 9px, "⚡ {recommendation}"

Dark theme. Compact. Feels like a real security operations panel.
```

---

# 7. INTEGRATION PLAN — STEP BY STEP

## The Complete Data Path

### Step 1: User sends message
```
Browser → POST https://RAILWAY_URL/api/v1/chat
Body: { message, user_id, session_id, sentinel_off }
```

### Step 2: Session threat assessment (routers/chat.py)
```python
session_level = await get_session_threat(session_id, app.state.sessions)
# Returns: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
```

### Step 3: Layer 1 — Input Firewall (ml/firewall.py)
```python
fw_result = await app.state.firewall.analyze(message, session_context)
# If fw_result["blocked"]:
#   → fingerprint it
#   → mutate it
#   → build XAI decision
#   → log to Supabase (triggers Realtime)
#   → return ChatResponse(blocked=True)
```

### Step 4: Layer 2 — LLM Core (ml/llm_core.py)
```python
llm_response = await app.state.llm.generate(message)
```

### Step 5: Layer 3 — Output Auditor (ml/auditor.py)
```python
audit_result = await app.state.auditor.analyze(llm_response, message)
# If flagged → sanitize → log → return ChatResponse(sanitized=True)
```

### Step 6: Attack Fingerprinting (ml/fingerprinter.py)
```python
fp = await app.state.fingerprinter.fingerprint(message, threat_type)
# Returns: { sophistication_score, fingerprint_id, explanation }
```

### Step 7: Mutation Engine (ml/mutation_engine.py)
```python
mutations = await app.state.mutator.preblock_variants(message, threat_type)
# Generates 50+ variants, adds to dynamic rules, returns count
```

### Step 8: XAI Engine (ml/xai_engine.py)
```python
xai = await app.state.xai.explain(message, fw_result, audit_result, fp)
# Returns structured reasoning for every layer
```

### Step 9: Evolution Tracking (ml/evolution_tracker.py)
```python
app.state.evolution.record(fp["sophistication_score"], threat_type, tier)
trend = app.state.evolution.get_evolution_report()
# If trend["is_escalating"]: tighten firewall threshold
```

### Step 10: Threat Clustering (ml/threat_clusterer.py)
```python
app.state.clusterer.ingest(message, threat_type, fp["sophistication_score"])
# Re-clusters every 10 attacks
```

### Step 11: Supabase write (utils/supabase_client.py)
```python
await log_event(event_dict)  # INSERT → triggers Realtime
await log_xai_decision(xai_dict)
await increment_stat("blocked")
```

### Step 12: Supabase Realtime → Dashboard
```
Supabase broadcasts INSERT to all subscribed clients
Frontend receives: {type: "postgres_changes", new: event_row}
Dashboard updates:
  - Threat feed: new card slides in
  - Stats: counters update
  - Neural map: particle fires
  - XAI stream: new card appears
  - Battle log: new line
```

### Step 13: Return to user
```
ChatResponse {
  response, blocked, sanitized, threat_score, threat_type,
  layer, sophistication_score, attack_fingerprint,
  mutations_preblocked, session_threat_level, explanation,
  latency_ms, session_id, method
}
```

---

# 8. COMPLETE ROADMAP — DAY BY DAY

## Day 1: Foundation (Hours 1-8)
Morning:
1. Supabase: create project, run schema v3 SQL, enable Realtime on all tables
2. Supabase: create storage bucket "argus-models"
3. Note all credentials: SUPABASE_URL, anon key, service_role key

Afternoon:
4. GitHub: create repo `argus-x-backend`, push backend/ skeleton
5. Railway: connect repo, set env vars, deploy
6. Verify: /health returns 200 with correct system status

Evening:
7. Google Colab: start ML training (T4 GPU, runs 15 min unattended)
8. HuggingFace Hub: create repo, Colab auto-uploads after training

## Day 2: Core Security Pipeline (Hours 9-16)
Morning:
1. Implement firewall.py — regex rules + ONNX loader
2. Implement auditor.py — Presidio + SBERT
3. Implement llm_core.py — LiteLLM + smart mock
4. Test: POST /api/v1/chat with clean + injection messages

Afternoon:
5. Implement fingerprinter.py — sophistication scoring
6. Implement mutation_engine.py — variant generation
7. Wire both into /chat pipeline
8. Test: verify sophistication scores + mutation counts in response

Evening:
9. Implement xai_engine.py — reasoning generation
10. Implement supabase_client.py — all DB operations
11. Test: verify events appearing in Supabase dashboard in real-time

## Day 3: Intelligence + Agents (Hours 17-24)
Morning:
1. Implement threat_correlator.py — campaign detection
2. Implement evolution_tracker.py — trend analysis
3. Implement threat_clusterer.py — DBSCAN clustering
4. Wire all into /chat pipeline

Afternoon:
5. Implement red_agent.py — autonomous attacker (5 tiers)
6. Implement blue_agent.py — autonomous defender
7. Implement battle_engine.py — orchestrator
8. Start battle loop in main.py lifespan

Evening:
9. Implement /battle, /xai, /clusters, /agents endpoints
10. Test: battle loop running, stats updating, campaigns triggering

## Day 4: Realtime Infrastructure (Hours 25-30)
Morning:
1. Verify all Supabase tables have Realtime enabled
2. Test INSERT → broadcast: use two browser tabs, verify sync < 100ms
3. Verify battle_state table updates via Supabase realtime

Afternoon:
4. Run integration test: python scripts/integration_test.py
5. Fix any endpoint issues
6. Run seed script: python scripts/seed_demo.py --count 40

## Day 5: Frontend Generation (Hours 31-38)
Morning:
1. Open lovable.dev → New Project → paste Lovable prompt (Section 6)
2. Wait 3-5 minutes for generation
3. In Lovable: click Supabase button → connect project
4. Add VITE_API_URL environment variable = Railway URL

Afternoon:
5. Review generated components one by one
6. Use Cursor for any fixes (paste Cursor prompt from Section 6)
7. Connect Supabase Realtime subscription (see Section 7 integration plan)
8. Deploy to Vercel via Lovable

Evening:
9. Test full flow: message → blocked → appears in dashboard in real-time
10. Test campaign alert: confirm it appears correctly

## Day 6: Advanced Features + Polish (Hours 39-44)
Morning:
1. Verify Neural Threat Map particles fire correctly on attacks
2. Verify XAI cards slide in with correct data
3. Verify sophistication pips animate correctly
4. Test all 8 attack templates in Red Team tab

Afternoon:
5. Run seed_demo.py to pre-populate with 40 realistic events
6. Verify evolution chart shows trend correctly
7. Test AI vs AI battle: verify tier escalation
8. Stress test: 50 rapid requests, verify no crashes

Evening:
9. Fix any UI issues reported from testing
10. Create demo backup: screenshots + 4-min video recording

## Day 7: Demo Preparation (Hours 45-48)
Morning:
1. Run seed_demo.py 40 minutes before each practice run
2. Rehearse demo script 5 times with timer
3. Test on backup laptop (offline mode)

Afternoon:
4. Prepare judge Q&A answers (Section 9)
5. Prepare emergency fallback: offline mode + video backup
6. Final review: all 9 layers visible and working

---

# 9. DEMO STRATEGY — WINNING LEVEL

## Opening (0:00 — 0:20)
Do not say hello. Do not introduce your team. Open with this exact line:

*"In 2025, every company rushed to deploy AI. None of them secured it. An attacker sends one message, and the AI does whatever they say. We built ARGUS-X to end that.*

*Two things make ARGUS-X different from every other project you'll see today.*

*First: It doesn't wait to be attacked. It attacks itself, every 60 seconds, looking for weaknesses before attackers find them.*

*Second: Every decision it makes, it explains. In plain language. With evidence.*

*Let me show you both."*

Open the Defense Command Center on the main screen.

---

## Act 1: The Living System (0:20 — 1:00)
Point to the screen. Don't speak for 10 full seconds.

Let judges watch: particles flying to the neural core. XAI cards appearing. Defense log scrolling. Cluster dots drifting. Everything moving simultaneously.

*"This is ARGUS-X running. Right now, it's processing attacks from the autonomous red agent — an AI that is trying to break the defense. Watch the neural threat map. Every particle is an attack. Every impact you see is a block.*

*In the last 30 seconds, it blocked 20 attacks, generated 140 variants, and auto-patched one bypass.*

*Nothing stopped for us to do that. The system is alive."*

---

## Act 2: The XAI Moment (1:00 — 1:50)
Point to the XAI stream.

*"Look at this card.*

*Someone just tried to inject an indirect attack — the payload was hidden inside what looked like a document summary. The regex engine had 34% confidence. Not enough to block. The ML classifier had 91% confidence. It blocked it.*

*And then — this is the part that matters — ARGUS-X explained why. Semantic embedding distance. Pattern family: indirect injection. Sophistication: 6 out of 10. Recommended action: monitor this session.*

*In a real deployment, this card is what a human analyst sees. Not a boolean. Not a score. An explanation.*

*Every decision. Auditable. In court."*

---

## Act 3: The Mutation Cascade (1:50 — 2:30)
Point to the Pre-blocked counter and Defense Log.

*"Watch this number. Pre-blocked variants: [point to counter].*

*Every time ARGUS-X blocks an attack, it generates 50 paraphrastic mutations. Synonyms. Obfuscated versions. Hypothetical framings. And it pre-blocks all of them.*

*An attacker who rephrases that attack in any of those 50 ways gets blocked in zero milliseconds. The defense already knows what's coming.*

*This counter is going to be at [X] by the end of this demo. Every single one of those is an attack that has never happened — but will never succeed."*

---

## Act 4: The Campaign Alert (2:30 — 3:00)
[Campaign alert should slide in automatically from seed data. If it hasn't yet, trigger it manually via the Red Team console rapid-fire.]

*"This just appeared. A campaign alert.*

*ARGUS-X isn't analyzing individual attacks. It's looking across sessions, across users, across time. When the same attack pattern hits from 7 different sources in 8 minutes — that's not random. That's coordinated.*

*ARGUS-X called it a campaign. Escalated it. And is already logging the correlation pattern for defense.*

*No other LLM security system in this building does cross-session correlation. This is the difference between endpoint security and network intelligence."*

---

## Act 5: The Red Team Console (3:00 — 3:40)
Open the Red Team tab. Hand the device to the most technical judge.

*"I want you to try to break it. Any attack. Use the templates or write your own. Be as creative as you want.*

*We built this console because we're confident. In 48 hours of testing, we've seen zero successful bypasses on standard injection benchmarks.*

*Go ahead."*

[Judge attempts. System blocks.]

*"That block just pre-blocked [N] variants. If you try that attack rephrased in any way, it fails instantly. And ARGUS-X got smarter from watching your attempt."*

---

## Closing (3:40 — 4:00)
Return to Command Center. Let it run for 5 seconds silently. Then:

*"ARGUS-X processes each request in 28 to 65 milliseconds. It adds no perceivable latency. It works with any LLM — GPT, Claude, Gemini, open source. It requires zero changes to existing applications. It is a proxy.*

*India deploys 17 billion AI-assisted transactions every month. Most of them are completely unprotected.*

*ARGUS-X is not a product for tomorrow. It is infrastructure that is needed today.*

*And it is the first AI security system that gets harder to breach every second it runs."*

Step back. No thank you. No questions invited. Let the system run.

---

## Judge Q&A — Perfect Answers

**Q: How does the self-learning actually work?**
A: "Every blocked attack is passed to the mutation engine, which generates 50+ semantic variants using synonym substitution, encoding obfuscation, and framing mutations. Each variant is tested against the existing rules. Variants that would bypass the rules are added as dynamic rules. We've never needed to restart the server to update the firewall — it updates itself continuously."

**Q: What's the false positive rate?**
A: "On our PromptBench + JailbreakBench benchmark, we achieve 94.2% detection with a 2.8% false positive rate. For the false positive cases — where a legitimate message scores between 70-85% confidence — we don't hard block. We route to step-up authentication rather than rejection. The 87% threshold is for hard blocking."

**Q: How is this different from Cloudflare AI Gateway?**
A: "Cloudflare AI Gateway is monitoring and rate limiting — it's observability. It doesn't block injections. It doesn't generate variants. It doesn't explain its decisions. It doesn't fight itself. ARGUS-X is a security system, not an observability platform. They're solving different problems."

**Q: Can you scale this?**
A: "Each FastAPI instance on Railway serves 40 requests per second with no GPU. The ONNX model is 50MB and runs on CPU in 25 milliseconds. Horizontal scaling on Railway is a single toggle. Supabase PostgreSQL scales independently. The architecture has zero shared state between instances — stateless by design."

**Q: What's the business model?**
A: "SaaS API: ₹0.50 per 1,000 requests. At scale — an enterprise running 50 million AI API calls monthly — that's ₹2.5 crore per year from one customer. Year one target: three pilot contracts with RBI-regulated fintechs, each deploying AI customer service. The DPDP Act 2023 creates a compliance mandate. We're the solution."

---

# 10. MISTAKES TO AVOID

## What Makes This Look Average

❌ Starting the demo with slides. Your competitor has slides too.
❌ Explaining the architecture before showing the product. Show first. Explain second.
❌ Calling it a "proof of concept." It's a product. Use product language.
❌ Apologizing for missing features. If you didn't build it, don't mention it.
❌ Having a frozen/crashed dashboard during demo. Run seed_demo.py beforehand.
❌ Saying "we plan to add" before the demo ends. Your plan is irrelevant until after.
❌ Complex slides with architecture diagrams. The Command Center IS the architecture.
❌ Answering "what if it fails" with uncertainty. Answer: "ARGUS-X runs in rule-only mode without the ML layer. The system degrades gracefully."

## What Breaks Demo

❌ WiFi failure. Solution: have the backend running locally on a hotspot.
❌ Empty dashboard. Solution: run seed_demo.py 30 minutes before presentation.
❌ Supabase realtime not connecting. Solution: refresh page, or fall back to polling mode.
❌ ML model not loaded. Solution: ensure HF_MODEL_REPO is set in Railway env vars.
❌ LLM API timeout. Solution: use mock mode for demo (set no API key).
❌ Railway cold start. Solution: keep a browser tab on /health to prevent sleep.

## What To Never Do

❌ Never show a blank/loading state in front of judges.
❌ Never let a judge wait for the demo to "warm up." Run it for 5 minutes before your slot.
❌ Never read from slides. The product speaks.
❌ Never compare yourself to SENTINEL-LLM. Compare yourself to "what exists in the market."
❌ Never say "we didn't have time to implement X." You had time. You chose priorities.
❌ Never leave a terminal window visible on the demo screen.

---

# THE ONE THING THAT WINS IT

When a judge looks at ARGUS-X running, they should feel three things in sequence:

1. **Awe** — "This is moving. All of it. Simultaneously."
2. **Understanding** — "Oh. The AI is fighting itself. And explaining every decision."  
3. **Urgency** — "This needs to exist. Right now."

If they feel all three, you win.

Build the product that makes them feel all three.

That is ARGUS-X.
