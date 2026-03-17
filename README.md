# 🏭 SmartChain Agentic AI

> **Intelligent Supply Chain Multi-Agent System — Forecasting · Risk Intelligence · RAG Chatbot · Autonomous Monitoring**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://tensorflow.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co)
[![LangChain](https://img.shields.io/badge/LangChain-Agents-purple)](https://langchain.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://smartchain-ai-i82g936efdbyvbqwd2wume.streamlit.app)

---

## 🎯 Problem Statement

Companies lose **$8 billion+** annually due to supply chain disruptions.
Most still rely on Excel sheets with zero AI intelligence.

**SmartChain Agentic AI** gives any company a fully autonomous AI brain that:
- 🤖 **Orchestrates** multiple specialized AI agents automatically
- 📈 **Predicts** demand surges using LSTM deep learning
- ⚠️ **Detects** supply chain risks from news using BERT NLP
- 🔍 **Answers** supply chain questions using RAG (FAISS + RoBERTa)
- 👁️ **Monitors** risks autonomously every 60 seconds — no human trigger needed
- 💾 **Remembers** past decisions using persistent agent memory

---

## 🧠 Agentic Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User / Dashboard                         │
│              Streamlit UI  ·  REST API                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              🧠 Orchestrator Agent                          │
│     Routes query · ReAct logic · LangChain Tools            │
└────┬──────────────┬──────────────┬──────────────┬───────────┘
     │              │              │              │
┌────▼────┐  ┌──────▼──────┐ ┌────▼────┐  ┌──────▼──────┐
│📈 Fore- │  │ ⚠️  Risk    │ │🔍 RAG   │  │ 📋 Report  │
│cast     │  │ Agent       │ │Agent    │  │ Agent      │
│LSTM+CNN │  │ DistilBERT  │ │RoBERTa  │  │ Combines   │
│         │  │ 5 categories│ │+ FAISS  │  │ all agents │
└─────────┘  └─────────────┘ └─────────┘  └────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                  💾 Agent Memory (SQLite)                   │
│         Session history · risk alerts · decision log        │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│              👁️  Autonomous Monitor Agent                   │
│       Scans news every 60s · no human trigger needed        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Full Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Orchestrator | LangChain AgentExecutor + ReAct | Routes queries to right agent |
| Demand Forecasting | LSTM + 1D-CNN (TensorFlow) | Predict future demand |
| Risk Detection | DistilBERT (HuggingFace) | Classify supply chain risks |
| RAG Chatbot | RoBERTa + FAISS + LangChain | Knowledge base Q&A |
| Agent Memory | SQLite + SQLAlchemy | Persistent decision history |
| Autonomous Monitor | Python schedule library | Background risk scanning |
| Backend API | FastAPI + Uvicorn | 8 REST endpoints |
| Frontend | Streamlit + Plotly | 5-page interactive dashboard |
| Deployment | Docker + Streamlit Cloud | Containerized + live URL |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/AKASHRAUT108/smartchain-ai.git
cd smartchain-ai
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Add your NEWSAPI_KEY and HF_TOKEN
```

### 5. Build knowledge base + agent memory
```bash
python src/rag/ingest.py
```

### 6. Start everything (4 terminals)
```bash
# Terminal 1 — API
uvicorn src.api.main:app --reload --port 8000

# Terminal 2 — Dashboard
streamlit run dashboard/app.py

# Terminal 3 — Autonomous Monitor
python -m src.agents.monitor

# Terminal 4 — Test Orchestrator
python -m src.agents.orchestrator
```

Open → `http://localhost:8501`

---

## 🐳 Docker Deployment

```bash
docker-compose up --build
```

- Dashboard → `http://localhost:8501`
- API Docs → `http://localhost:8000/docs`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | System health check |
| POST | `/forecast` | LSTM demand prediction |
| POST | `/risk` | BERT risk classification |
| GET | `/risk-feed` | Live news risk scores |
| POST | `/chat` | RAG chatbot Q&A |
| POST | `/agent/run` | Orchestrator agent (auto-routes) |
| GET | `/agent/alerts` | Autonomous risk alerts |
| GET | `/agent/memory/{id}` | Agent memory for session |

---

## 🤖 Agent Capabilities

| Agent | Trigger Keywords | Model |
|---|---|---|
| DemandForecaster | demand, forecast, inventory, stock | LSTM + 1D-CNN |
| RiskDetector | strike, disaster, shortage, tariff, war | DistilBERT |
| SupplyChainKnowledge | how do, what is, calculate, KPI | RoBERTa + FAISS |
| StatusReport | status, report, overview, summary | All agents combined |

---

## 📊 Model Performance

| Model | Metric | Score |
|---|---|---|
| LSTM Demand | MAE | ~23 units/day |
| BERT Risk | Accuracy | ~83% (5 categories) |
| RoBERTa QA | Avg Confidence | ~85% |
| Orchestrator | Routing Accuracy | ~95% (keyword scoring) |

---

## 📁 Project Structure

```
smartchain-ai/
├── src/
│   ├── agents/
│   │   ├── tools.py          # LangChain Tool wrappers
│   │   ├── orchestrator.py   # Master routing agent + memory
│   │   └── monitor.py        # Autonomous background monitor
│   ├── forecasting/          # LSTM training + prediction
│   ├── risk_nlp/             # BERT fine-tuning + inference
│   ├── rag/                  # FAISS + RoBERTa QA chatbot
│   └── api/                  # FastAPI backend (8 endpoints)
├── dashboard/                # Streamlit 5-page frontend
├── data/
│   ├── raw/                  # Original datasets
│   ├── processed/            # Features + FAISS + agent DB
│   └── knowledge_base/       # Supply chain documents
├── models/                   # Saved model files
├── notebooks/                # EDA + experiments
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🌐 Live Demo

**[smartchain-ai-i82g936efdbyvbqwd2wume.streamlit.app](https://smartchain-ai-i82g936efdbyvbqwd2wume.streamlit.app)**

---

## 👤 Author

**Akash Raut**
- GitHub: [@AKASHRAUT108](https://github.com/AKASHRAUT108)
- Built as a fresher ML portfolio project — March 2026
- Open to ML/AI/Data Science roles
