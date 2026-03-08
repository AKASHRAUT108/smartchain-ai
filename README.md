# 🏭 SmartChain AI

> **Intelligent Supply Chain Risk Intelligence & Demand Forecasting System**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://tensorflow.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)

---

## 🎯 Problem Statement

Companies lose **$8 billion+** annually due to supply chain disruptions.
Most still rely on Excel sheets with zero AI intelligence.

**SmartChain AI** gives any company a real-time AI brain that:
- 📈 Predicts demand surges using LSTM deep learning
- ⚠️ Detects supply chain risks from news using BERT NLP
- 🤖 Answers supply chain questions using RAG (Retrieval-Augmented Generation)
- 🚨 Flags shipment anomalies using 1D-CNN

---

## 🧠 Architecture
```
┌─────────────────────────────────────────────────┐
│           SmartChain AI Dashboard               │
│              (Streamlit UI :8501)               │
└────────────┬──────────────┬───────────────────┬─┘
             │              │                   │
    ┌────────▼──────┐ ┌─────▼──────┐  ┌────────▼──────┐
    │  LSTM Demand  │ │  BERT Risk │  │  RAG Chatbot  │
    │  Forecasting  │ │  Classifier│  │  RoBERTa QA   │
    └───────────────┘ └────────────┘  └───────────────┘
             │              │                   │
    ┌────────▼──────────────▼───────────────────▼──────┐
    │              FastAPI Backend (:8000)              │
    │     /forecast  /risk  /risk-feed  /chat  /health  │
    └───────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Demand Forecasting | LSTM + 1D-CNN (TensorFlow) |
| Risk Detection | DistilBERT (HuggingFace Transformers) |
| RAG Chatbot | RoBERTa + FAISS + LangChain |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit + Plotly |
| Deployment | Docker + Docker Compose |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/smartchain-ai.git
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

### 5. Build the knowledge base
```bash
python src/rag/ingest.py
```

### 6. Start the API
```bash
uvicorn src.api.main:app --reload --port 8000
```

### 7. Start the Dashboard (new terminal)
```bash
streamlit run dashboard/app.py
```

Open → `http://localhost:8501`

---

## 🐳 Docker Deployment
```bash
docker-compose up --build
```

- Dashboard → `http://localhost:8501`
- API Docs  → `http://localhost:8000/docs`

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | System health check |
| POST | `/forecast` | LSTM demand prediction |
| POST | `/risk` | BERT risk classification |
| GET | `/risk-feed` | Live news risk scores |
| POST | `/chat` | RAG chatbot Q&A |

---

## 📊 Model Performance

| Model | Metric | Score |
|---|---|---|
| LSTM Demand | MAE | ~23 units |
| BERT Risk | Accuracy | ~83% |
| RoBERTa QA | Confidence | ~85% avg |

---

## 📁 Project Structure
```
smartchain-ai/
├── src/
│   ├── forecasting/    # LSTM training + prediction
│   ├── risk_nlp/       # BERT fine-tuning + inference
│   ├── rag/            # FAISS + RoBERTa QA chatbot
│   └── api/            # FastAPI backend
├── dashboard/          # Streamlit frontend
├── data/
│   ├── raw/            # Original datasets
│   ├── processed/      # Features + FAISS index
│   └── knowledge_base/ # Supply chain documents
├── models/             # Saved model files
├── notebooks/          # EDA + experiments
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 👤 Author

**Akash Raut**
- GitHub: [@AKASHRAUT108](https://github.com/AKASHRAUT108)
- Built as a fresher ML portfolio project — March 2026
```

---

## 🔐 Step 4 — Create .env.example

Create `.env.example` (safe to commit — no real keys):
```
NEWSAPI_KEY=your_newsapi_key_here
HF_TOKEN=your_huggingface_token_here
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True