# AI Chatbot Developer: Simulated Project Output

**Project:** Upwork Talent Accelerator - Customer Support Chatbot for Outdoor E-Commerce  
**Deliverable Type:** Custom Code Repository & Architecture Implementation  
**Brand:** North Star Support Bot

---

## Project Overview

This document describes the architecture and implementation of a customer support chatbot for an outdoor apparel and camping gear e-commerce business.

The solution uses an **agentic LangGraph workflow** with a **local intent classifier**, **hybrid offline RAG**, and **LangChain prompt templates**. Every reply is grounded in SQLite / catalog data. The bot runs **fully offline** — no LLM provider APIs, no API keys, and no external services are required for evaluators to review it.

---

## System Architecture & Tech Stack

The system is split into a React frontend, a FastAPI + LangGraph backend, and a local data layer.

### 1. Agentic Orchestration (Python & LangGraph)

A real LangGraph `StateGraph` manages the stateful multi-actor workflow:

```
START -> ingest -> [live_agent | router]
router -> order | returns | shipping | recommendations | handoff | menu | fallback
every agent -> respond -> END
```

- **Router (local ML):** A scikit-learn TF-IDF + LogisticRegression classifier trained in-process from a seed corpus of labelled phrasings. High-precision deterministic overrides (explicit order numbers, "talk to a human", "main menu", active slot-filling) always win. Low-confidence or out-of-vocabulary input routes to `fallback` instead of guessing. A keyword router is the last-resort fallback if scikit-learn is unavailable.
- **Agent nodes:** Gather grounded facts only — order rows from SQLite, return/shipping policies from the seeded catalog, product picks from the local retriever.
- **Respond node (LangChain):** Renders `PromptTemplate`s via LCEL `Runnable` chains, then runs a **grounding guard**. If a reply drops or contradicts a retrieved order status, the deterministic fact string is emitted instead.

### 2. Data & Retrieval Layer (Offline RAG)

- **Structured data (SQLite):** Mock orders, products, and FAQs seeded from `backend/app/data/catalog.json`. Orders `#111` / `#222` / `#333` match the contract exactly. Postgres via Docker is optional and unused by default.
- **Unstructured retrieval (local hybrid index):** TF-IDF cosine similarity + BM25 over the product catalog and FAQs, exposed as a LangChain `BaseRetriever`. The fitted index is cached to `backend/app/data/retriever_index.joblib` and rebuilt when the catalog changes. No hosted vector DB and no remote embeddings.

### 3. API & Frontend UI

- **Backend API:** FastAPI bridges the LangGraph workflow and the UI (`POST /chat`, `POST /chat/reset`, `GET /chat/{session_id}`, `GET /health`).
- **Frontend:** React + TypeScript chat interface with quick replies and a simulated Live Agent banner.

---

## Core Use Case Implementation

### 1. Order Tracking (Structured Data)

- **Flow:** The router detects an order-status intent (or an explicit order number) and selects the `order_agent`.
- **Execution:** The system extracts the order number and queries SQLite.
- **Resolution:** A LangChain template formats the grounded result, e.g. _"Order #111 (Alpine Ridge Tent) is shipped - arriving tomorrow."_
- **Contract data:** `#111` Shipped / arriving tomorrow · `#222` Processing / ships in 24 hours · `#333` Delivered · any other → invalid.

### 2. Returns and Exchanges (Hybrid Processing)

- **Flow:** The user requests a return or asks about the return policy.
- **Execution:** The system loads the 30-day / unused / original-packaging policy from the catalog, retrieves the matching FAQ via the local retriever, and optionally checks whether a named order is still inside the return window.
- **Resolution:** The bot answers with the policy and a mock returns link (`https://northstar.example/returns`). Eligible orders get a mock return-label URL.

### 3. Product Recommendations (Offline RAG)

- **Flow:** User asks something like _"What's the best sleeping bag for sub-zero temperatures?"_ The router selects the `recommendation_agent`.
- **Execution:** The agent asks 1–2 clarifying questions when needed, then runs a hybrid TF-IDF + BM25 similarity search over the outdoor gear catalog.
- **Resolution:** Templates present 2–3 ranked recommendations with grounded specs (e.g. Summit Down Sleeping Bag, rated to -20°F, 800-fill goose down).

### 4. Human Handoff (State Interruption)

- **Flow:** The user says _"Talk to a human"_ / similar phrasing, or hits repeated fallback.
- **Execution:** LangGraph sets `mode` to `live_agent`, pausing the AI agent path.
- **Resolution:** The frontend shows a Live Agent banner. The transcript stays in session state. Saying _"main menu"_ (or a clear menu request) returns the user to the bot.

### 5. Fallback

Clear "I didn't understand" response, quick-reply options, and escalation to a live agent after a second miss.

---

## Setup & Execution (For Evaluators)

No API keys. No Docker required. No accounts or paid subscriptions.

### One-command start

```powershell
# Windows PowerShell
.\run.ps1
# If scripts are disabled:  .\run.cmd   or   powershell -ExecutionPolicy Bypass -File .\run.ps1
```

```bash
# macOS / Linux / Git Bash
./run.sh
```

Then open http://127.0.0.1:5173

### Manual start

Prerequisites: Python 3.10–3.12 and Node.js 18+.

```bash
# Backend
cd backend
py -3.11 -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python seed_mock_data.py
uvicorn app.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Health check: http://localhost:8000/health  
Expected shape: `engine: local`, `router: sklearn-tfidf-logreg`, `retriever: tfidf-cosine+bm25`.

### Tests

```bash
cd backend
python smoke_test.py      # four use cases + fallback
python offline_test.py    # router paraphrases, retrieval ranking, grounding guard
```

Both suites run with no keys and no network.

### Optional configuration

Copy `.env.example` to `backend/.env` only if you want to tune CORS, the intent confidence threshold, or the retrieval score floor. There are no API key variables.

`docker compose --profile full up -d postgres` is optional and unused by the default SQLite path.
