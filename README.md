# North Star Support Bot

Customer support chatbot for an outdoor apparel & camping gear store (Upwork Talent Accelerator simulation).

**Runs fully offline. No API keys, no LLM providers, no external services, no network calls.**
All intent recognition and retrieval is computed locally with scikit-learn, and every reply is
rendered from SQLite/catalog data through LangChain prompt templates. Clone, run, done.

## Stack

- **Frontend:** React + TypeScript (Vite)
- **Backend:** FastAPI + a real **LangGraph** `StateGraph` (router -> database / hybrid / RAG / escalation / fallback -> response)
- **Intent router:** local **scikit-learn** TF-IDF + LogisticRegression classifier trained in-process from a seed corpus, behind high-precision deterministic overrides
- **Retrieval (RAG):** local hybrid **TF-IDF cosine + BM25** index over the product catalog and FAQs, exposed as a LangChain retriever
- **Response generation:** **LangChain** `PromptTemplate` rendering composed with LCEL `Runnable` chains, plus a grounding guard
- **Data:** SQLite (orders, products, FAQs) seeded from `backend/app/data/catalog.json`

## How it works

`POST /chat` calls `run_turn(state, message)`, which invokes the compiled LangGraph app:

```
START -> ingest -> [conditional] mode == live_agent ? live_agent : router

live_agent -> [conditional] clear menu request ? menu_agent : respond

router     -> [conditional] intent ?
                order_tracking   -> order_agent           (SQLite order lookup)
                returns          -> returns_agent         (policy + FAQ retrieval + order check)
                shipping         -> shipping_agent        (catalog shipping policy)
                recommendations  -> recommendation_agent  (hybrid RAG over the catalog)
                handoff          -> escalation_agent      (simulated live agent)
                menu             -> menu_agent
                fallback         -> fallback_agent        (escalates on the 2nd miss)

every agent node -> respond -> END
```

- **`router`** classifies the message into exactly one of `order_tracking`, `returns`, `shipping`,
  `recommendations`, `handoff`, `menu`, `fallback`. Layer 1 is deterministic overrides (an explicit
  order number, explicit "talk to a human", explicit "main menu", and active slot-filling context
  always win). Layer 2 is the local classifier, which generalises to unseen phrasings such as
  *"my parcel hasn't turned up"*, *"I'd like my money back"*, *"how quickly can you get it to me"*,
  and *"put me through to someone"*. If the top-class probability is below the confidence threshold,
  or the message contains no known vocabulary, the turn routes to `fallback` instead of guessing.
  Layer 3 is the original keyword router, used only if scikit-learn is unavailable so the bot never
  crashes.
- **Agent nodes** gather grounded facts only: orders come from SQLite, the 30-day/unused/original-packaging
  return policy and the 3-5 / 1-2 business day shipping times come from the seeded catalog, and product
  picks come from the local hybrid retriever.
- **`respond`** renders the selected `PromptTemplate` and then runs a **grounding guard**: if a reply
  ever drops or contradicts the retrieved order status, the deterministic reply is emitted instead.
  Nothing is paraphrased by a model, so a business fact can only appear if the data layer supplied it.

## Prerequisites

- **Python 3.10-3.12** (Python 3.13+ may lack prebuilt wheels on Windows)
- **Node.js 18+**

## Quick start (one command)

Both scripts create the virtualenv, install dependencies, seed the database, pick a free port, and start both servers. Press Ctrl+C to stop everything.

**Windows (PowerShell):**

```powershell
.\run.ps1
```

If PowerShell says scripts are disabled, either run:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

or double-click / run `.\run.cmd` instead (same launcher, bypasses the policy for that one run).

To permanently allow your own scripts (recommended once per machine):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Then `.\run.ps1` works normally.

**macOS / Linux / Git Bash:**

```bash
./run.sh
```

Then open http://127.0.0.1:5173

To run the test suite instead of the servers, add `-Smoke` (PowerShell) or `--smoke` (bash).

## Manual start

### 1. Backend (terminal 1)

```bash
cd backend
py -3.11 -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
python seed_mock_data.py
uvicorn app.main:app --reload --port 8000
```

Verify: http://localhost:8000/health

### 2. Frontend (terminal 2)

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## Ports

- Frontend: http://localhost:5173
- Backend: http://localhost:8000

If port 8000 is blocked on your machine, run the backend on 8080 instead:

```bash
uvicorn app.main:app --reload --port 8080
```

Then create `frontend/.env` with this line and restart the frontend:

```
VITE_API_URL=http://127.0.0.1:8080
```

## Demo scripts to try

| Use case | Try saying |
|----------|------------|
| Order tracking | `Where is my order?` then `#111` / `#222` / `#333` |
| Returns | `What is your return policy?` or `I want to return order 111` |
| Shipping | `How long is standard shipping?` |
| Recommendations | `Best sleeping bag for sub-zero temperatures` |
| Human handoff | `Talk to a human` then `main menu` to return |
| Fallback | `asdfgh qwerty` |

Then try the same intents in phrasings the bot has never seen, to check the local classifier rather
than a keyword list: `my parcel hasn't turned up`, `I'd like my money back`,
`how quickly can you get it to me`, `I need something for freezing nights`,
`put me through to someone`.

### Mock orders (exact)

- `#111` -> Shipped, arriving tomorrow
- `#222` -> Processing, ships in 24 hours
- `#333` -> Delivered
- Any other -> invalid order

### Policies (exact)

- Returns: 30-day window, unused items, original packaging -> https://northstar.example/returns
- Shipping: Standard 3-5 business days; Expedited 1-2 business days

## Tests

Both suites run offline with no keys, no network, and no services.

```bash
cd backend

# All four use cases plus fallback, end to end
python smoke_test.py

# Router generalisation, deterministic overrides, low-confidence fallback,
# retrieval ranking, LangGraph wiring, and the grounding guard
python offline_test.py
```

## Health check

`GET /health` reports which local components are live:

```json
{
  "status": "ok",
  "app": "North Star Support Bot",
  "engine": "local",
  "router": "sklearn-tfidf-logreg",
  "retriever": "tfidf-cosine+bm25",
  "graph": { "framework": "langgraph", "nodes": ["ingest", "live_agent", "router", "..."] }
}
```

`router` degrades to `keyword-fallback` and `retriever` to `keyword` if scikit-learn or `rank_bm25`
cannot be imported. `engine` is always `local`.

## Configuration

Everything has a working default, so `backend/.env` is optional. Copy `.env.example` to
`backend/.env` only if you want to change one of these:

| Variable | Default | Purpose |
|----------|---------|---------|
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Allowed browser origins |
| `INTENT_CONFIDENCE_THRESHOLD` | `0.30` | Minimum classifier probability before an intent is trusted |
| `RETRIEVAL_SCORE_FLOOR` | `0.05` | Minimum retrieval score for a catalog document to count as a match |

There are no API keys of any kind.

The trained classifier and the fitted retriever index are cached to
`backend/app/data/*.joblib` and rebuilt automatically whenever the seed corpus or catalog changes.

`docker-compose.yml` ships an **optional** Postgres profile for anyone who wants it. It is not
required and is not used by default; SQLite is the zero-setup default.

## Project layout

```
run.ps1 / run.sh                  One-command launchers (deps, seed, both servers)
backend/app/main.py               FastAPI endpoints (/chat, /chat/reset, /health)
backend/app/graph/workflow.py     LangGraph StateGraph: nodes, conditional edges, run_turn
backend/app/graph/state.py        ChatState channel schema
backend/app/agents/router.py      Overrides -> local classifier -> keyword fallback
backend/app/agents/classifier.py  TF-IDF + LogisticRegression, cached to disk
backend/app/agents/training_data.py  Seed corpus of labelled phrasings per intent
backend/app/rag/store.py          TF-IDF cosine + BM25 index as a LangChain retriever
backend/app/chains/prompts.py     LangChain PromptTemplates (all user-facing copy)
backend/app/chains/pipeline.py    LCEL chains + anti-hallucination grounding guard
backend/app/tools/                Order lookup + retrieval wrappers
backend/app/data/                 Seed catalog, SQLite db, cached model/index
backend/smoke_test.py             Four use cases + fallback
backend/offline_test.py           Router, retrieval, graph, and grounding tests
frontend/src/App.tsx              React chat UI + live-agent banner
upwork chatbot.md                 Architecture brief (matches this offline implementation)
AI Chatbot Contract.pdf           Requirements
```
