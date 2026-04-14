# Contributing to ARGUS-X

Thanks for your interest in ARGUS-X! Here's how to contribute effectively.

---

## 🏗️ Development Setup

```bash
# Clone and setup backend
git clone https://github.com/1harshkashyap/ARGUS_X.git
cd ARGUS_X/argus/backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r ../../requirements.txt
cp ../../.env.example .env

# Start backend
python main.py

# In a new terminal — setup frontend
cd ARGUS_X/argus/frontend
npm install
npm run dev
```

---

## 📐 Architecture Rules

1. **Backend modules are in `argus/backend/ml/`** — all ML/security logic lives here
2. **Routers are in `argus/backend/routers/`** — one file per API domain
3. **Frontend components are in `argus/frontend/src/components/`** — React + TypeScript
4. **All new routes** must be registered in `main.py` with `dependencies=[Depends(require_api_key)]`
5. **All user-facing strings** must be sanitized with `html.escape()` (backend) or `sanitizeText()` (frontend)

---

## 🔐 Security Requirements

These are **non-negotiable**:

- ❌ Never use `dangerouslySetInnerHTML` in React components
- ❌ Never log API keys, service keys, or user credentials
- ❌ Never expose `SUPABASE_SERVICE_KEY` to the frontend
- ✅ All new endpoints must go through `require_api_key` dependency
- ✅ All user inputs must have Pydantic validation with `max_length`
- ✅ All LLM/XAI output strings must be HTML-escaped before storage/display

---

## ✅ Verification Checklist

Before submitting a PR, verify:

```bash
# TypeScript compiles
cd argus/frontend && npx tsc --noEmit

# Python compiles
python -c "import py_compile; py_compile.compile('argus/backend/YOUR_FILE.py', doraise=True)"

# No secrets in frontend
grep -r "SERVICE_KEY" argus/frontend/  # Must return nothing
```

---

## 📝 Commit Style

```
fix: rate limiter creating new instance per call
feat: add org_id multi-tenant support to events
docs: update README with deployment guide
security: escape XAI output strings
```

---

## 🧪 Testing Attacks

Use the benchmark API to test without side effects:

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Your test prompt here"}'
```
