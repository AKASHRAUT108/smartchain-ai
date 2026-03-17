import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from agents.orchestrator import run_agent, init_memory_db, get_recent_alerts, get_memory_history
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import uvicorn
load_dotenv()

from risk_nlp.inference import predict_risk, analyze_news_feed
from rag.chatbot import load_rag_chain, ask

# ─── APP SETUP ─────────────────────────────────────────────
app = FastAPI(
    title       = "SmartChain AI",
    description = "Supply Chain Risk Intelligence & Demand Forecasting API",
    version     = "1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"]
)

# ─── LOAD ALL MODELS ON STARTUP ────────────────────────────
print("\n🚀 Loading all AI models...")

lstm_model = tf.keras.models.load_model(
    "models/lstm_demand_model/best_model.keras"
)
with open("data/processed/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
print("   ✅ LSTM demand model loaded")

retriever, tokenizer, qa_model = load_rag_chain()
print("   ✅ RAG chatbot loaded")

print("\n✅ All models ready — API is live!\n")

# ─── REQUEST SCHEMAS ───────────────────────────────────────
class ForecastRequest(BaseModel):
    days:             int = 7
    product_category: str = "all"

class ChatRequest(BaseModel):
    message:    str
    session_id: str = "default"

class RiskRequest(BaseModel):
    headline: str

# ─── ENDPOINTS ─────────────────────────────────────────────
init_memory_db()
print("   ✅ Agent memory initialized")

@app.get("/health")
def health():
    return {
        "status":  "ok",
        "models":  ["lstm_demand", "bert_risk", "rag_chatbot"],
        "version": "1.0.0"
    }

@app.get("/")
def root():
    return {"message": "Welcome to SmartChain AI 🚀 — visit /docs"}


@app.post("/forecast")
def forecast_demand(request: ForecastRequest):
    """Predict demand for next N days using LSTM."""
    try:
        X_test        = np.load("data/processed/X_test.npy")
        last_sequence = X_test[-1:]
        predictions   = []
        current_seq   = last_sequence.copy()

        for _ in range(request.days):
            pred = lstm_model.predict(current_seq, verbose=0)[0][0]
            predictions.append(float(pred))
            new_row               = current_seq[0, -1, :].copy()
            new_row[0]            = pred
            current_seq           = np.roll(current_seq, -1, axis=1)
            current_seq[0, -1, :] = new_row

        # Inverse scale
        dummy       = np.zeros((len(predictions), scaler.n_features_in_))
        dummy[:, 0] = predictions
        real_preds  = scaler.inverse_transform(dummy)[:, 0]
        real_preds  = np.maximum(real_preds, 0)

        return {
            "days_forecasted": request.days,
            "category":        request.product_category,
            "predictions":     [round(float(v), 1) for v in real_preds],
            "unit":            "units per day",
            "trend":           "increasing" if real_preds[-1] > real_preds[0]
                               else "decreasing"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/risk")
def analyze_risk(request: RiskRequest):
    """Classify supply chain risk from a news headline."""
    try:
        return predict_risk(request.headline)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/risk-feed")
def get_risk_feed():
    """Get analyzed risk scores from latest saved news feed."""
    try:
        news_path = "data/processed/live_news.csv"
        if not os.path.exists(news_path):
            return {
                "message": "No live news data yet. Run news_fetcher.py first.",
                "alerts":  []
            }
        news_df  = pd.read_csv(news_path)
        analyzed = analyze_news_feed(news_df)
        alerts   = analyzed[analyzed["is_risk"] == True].head(10)
        return {
            "total_articles": len(analyzed),
            "risk_alerts":    len(alerts),
            "alerts": alerts[[
                "title", "risk_category",
                "severity_score", "source"
            ]].to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
def chat(request: ChatRequest):
    """Ask SmartChain AI a supply chain question."""
    try:
        result = ask(retriever, tokenizer, qa_model, request.message)
        return {
            "question":   request.message,
            "answer":     result["answer"],
            "confidence": result["confidence"],
            "sources":    result["sources"],
            "session_id": request.session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class AgentRequest(BaseModel):
    query:      str
    session_id: str = "default"

@app.post("/agent/run")
def agent_run(request: AgentRequest):
    """Master Orchestrator Agent — routes query to best tool automatically."""
    try:
        result = run_agent(request.query, request.session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/alerts")
def agent_alerts():
    """Get all autonomous risk alerts detected by monitor agent."""
    try:
        alerts = get_recent_alerts(limit=20)
        return {
            "total_alerts": len(alerts),
            "alerts": [
                {
                    "timestamp": a[0],
                    "headline":  a[1],
                    "risk_cat":  a[2],
                    "severity":  a[3]
                }
                for a in alerts
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent/memory/{session_id}")
def agent_memory(session_id: str):
    """Get agent memory for a session."""
    try:
        history = get_memory_history(session_id, limit=10)
        return {
            "session_id": session_id,
            "history": [
                {
                    "timestamp": h[0],
                    "query":     h[1],
                    "tool_used": h[2],
                    "result":    h[3]
                }
                for h in history
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)