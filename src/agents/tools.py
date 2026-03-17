import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pickle
import tensorflow as tf
from langchain_core.tools import Tool
from risk_nlp.inference import predict_risk
from rag.chatbot import load_rag_chain, ask

print("📦 Loading models for agent tools...")

# ── Load LSTM ───────────────────────────────────────────────
lstm_model = tf.keras.models.load_model(
    "models/lstm_demand_model/best_model.keras"
)
with open("data/processed/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ── Load RAG ────────────────────────────────────────────────
retriever, tokenizer, qa_model = load_rag_chain()

print("✅ All agent tools ready\n")


# ── TOOL 1: Demand Forecasting ──────────────────────────────
def forecast_tool(query: str) -> str:
    """
    Forecasts product demand for next 7 days using LSTM model.
    Input: any string (query is ignored, uses latest data)
    Output: forecast summary string
    """
    try:
        X_test      = np.load("data/processed/X_test.npy")
        current_seq = X_test[-1:].copy()
        predictions = []

        for _ in range(7):
            pred = lstm_model.predict(current_seq, verbose=0)[0][0]
            predictions.append(float(pred))
            new_row               = current_seq[0, -1, :].copy()
            new_row[0]            = pred
            current_seq           = np.roll(current_seq, -1, axis=1)
            current_seq[0, -1, :] = new_row

        dummy       = np.zeros((7, scaler.n_features_in_))
        dummy[:, 0] = predictions
        real_preds  = scaler.inverse_transform(dummy)[:, 0]
        real_preds  = np.maximum(real_preds, 0)

        avg   = round(float(np.mean(real_preds)), 1)
        peak  = round(float(np.max(real_preds)), 1)
        trend = "INCREASING" if real_preds[-1] > real_preds[0] else "DECREASING"

        return (
            f"7-Day Demand Forecast:\n"
            f"  Average daily demand: {avg} units\n"
            f"  Peak demand: {peak} units\n"
            f"  Trend: {trend}\n"
            f"  Daily values: {[round(float(v),1) for v in real_preds]}"
        )
    except Exception as e:
        return f"Forecast error: {str(e)}"


# ── TOOL 2: Risk Detection ──────────────────────────────────
def risk_tool(headline: str) -> str:
    """
    Analyzes a news headline for supply chain risk.
    Input: news headline string
    Output: risk classification and severity
    """
    try:
        result = predict_risk(headline)
        if result["is_risk"]:
            return (
                f"⚠️ RISK DETECTED:\n"
                f"  Category:  {result['risk_category']}\n"
                f"  Severity:  {result['severity_score']}/10\n"
                f"  Confidence:{result['confidence']:.1%}\n"
                f"  Headline:  {headline}\n"
                f"  ACTION REQUIRED: Review supply chain for {result['risk_category']} impact"
            )
        else:
            return (
                f"✅ NO RISK: '{headline}'\n"
                f"  Classification: Normal supply chain news\n"
                f"  Confidence: {result['confidence']:.1%}"
            )
    except Exception as e:
        return f"Risk analysis error: {str(e)}"


# ── TOOL 3: RAG Knowledge Base ──────────────────────────────
def rag_tool(question: str) -> str:
    """
    Answers supply chain questions using knowledge base.
    Input: any supply chain question
    Output: answer with confidence score
    """
    try:
        result = ask(retriever, tokenizer, qa_model, question)
        return (
            f"Knowledge Base Answer:\n"
            f"  Question:   {question}\n"
            f"  Answer:     {result['answer']}\n"
            f"  Confidence: {result['confidence']:.1%}\n"
            f"  Sources:    {len(result['sources'])} chunks retrieved"
        )
    except Exception as e:
        return f"RAG error: {str(e)}"


# ── TOOL 4: Full Status Report ──────────────────────────────
def status_report_tool(query: str) -> str:
    """
    Generates a complete supply chain status report.
    Combines forecast + risk + knowledge base insights.
    Input: any string
    Output: full status report
    """
    forecast = forecast_tool("get forecast")
    risk1    = risk_tool("Port strike shipping delay disruption")
    kb       = rag_tool("What are the key KPIs for supply chain performance?")

    return (
        f"=== SMARTCHAIN AI STATUS REPORT ===\n\n"
        f"📈 DEMAND OUTLOOK:\n{forecast}\n\n"
        f"⚠️ RISK ASSESSMENT (sample):\n{risk1}\n\n"
        f"📚 KPI GUIDANCE:\n{kb}\n\n"
        f"=== END OF REPORT ==="
    )


# ── REGISTER LANGCHAIN TOOLS ────────────────────────────────
AGENT_TOOLS = [
    Tool(
        name        = "DemandForecaster",
        func        = forecast_tool,
        description = (
            "Use this tool to forecast product demand for the next 7 days. "
            "Call this when the user asks about demand, sales prediction, "
            "inventory planning, or future orders. "
            "Input can be any string describing what they want to forecast."
        )
    ),
    Tool(
        name        = "RiskDetector",
        func        = risk_tool,
        description = (
            "Use this tool to analyze supply chain risk from a news headline or event. "
            "Call this when user mentions: strike, disaster, shortage, tariff, war, "
            "disruption, port, typhoon, earthquake, or any geopolitical event. "
            "Input must be the news headline or event description."
        )
    ),
    Tool(
        name        = "SupplyChainKnowledge",
        func        = rag_tool,
        description = (
            "Use this tool to answer questions about supply chain best practices, "
            "KPIs, protocols, inventory management, logistics, and procedures. "
            "Call this when user asks 'how do I', 'what is', 'what should I do', "
            "or needs guidance from the knowledge base. "
            "Input must be the question string."
        )
    ),
    Tool(
        name        = "StatusReport",
        func        = status_report_tool,
        description = (
            "Use this tool to generate a complete supply chain status report. "
            "Call this when user asks for overall status, summary, dashboard, "
            "or a full report of current supply chain health. "
            "Input can be any string."
        )
    ),
]