import os
import sys
import json
import sqlite3
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Import our tools
from src.agents.tools import AGENT_TOOLS

# ── MEMORY DATABASE ─────────────────────────────────────────
DB_PATH = "data/processed/agent_memory.db"

def init_memory_db():
    """Create SQLite tables for agent memory."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS agent_memory (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            query       TEXT,
            tool_used   TEXT,
            result      TEXT,
            session_id  TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS risk_alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT,
            headline    TEXT,
            risk_cat    TEXT,
            severity    REAL,
            actioned    INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()
    print("✅ Agent memory database initialized")


def save_to_memory(query, tool_used, result, session_id="default"):
    """Save agent decision to memory."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute(
        "INSERT INTO agent_memory (timestamp, query, tool_used, result, session_id) VALUES (?,?,?,?,?)",
        (datetime.now().isoformat(), query, tool_used, str(result)[:500], session_id)
    )
    conn.commit()
    conn.close()


def get_memory_history(session_id="default", limit=5):
    """Get recent agent decisions for context."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute(
        "SELECT timestamp, query, tool_used, result FROM agent_memory WHERE session_id=? ORDER BY id DESC LIMIT ?",
        (session_id, limit)
    )
    rows = c.fetchall()
    conn.close()
    return rows


def save_risk_alert(headline, risk_cat, severity):
    """Save a detected risk alert."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute(
        "INSERT INTO risk_alerts (timestamp, headline, risk_cat, severity) VALUES (?,?,?,?)",
        (datetime.now().isoformat(), headline, risk_cat, severity)
    )
    conn.commit()
    conn.close()


def get_recent_alerts(limit=10):
    """Get recent risk alerts."""
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute(
        "SELECT timestamp, headline, risk_cat, severity FROM risk_alerts ORDER BY id DESC LIMIT ?",
        (limit,)
    )
    rows = c.fetchall()
    conn.close()
    return rows


# ── INTELLIGENT ROUTER ──────────────────────────────────────
# Since we're using local models, we build a smart keyword-based
# router instead of an LLM-powered router — same result, no API needed

TOOL_KEYWORDS = {
    "DemandForecaster": [
        "demand", "forecast", "predict", "sales", "inventory",
        "order", "units", "stock", "supply", "quantity", "how many"
    ],
    "RiskDetector": [
        "strike", "disaster", "shortage", "tariff", "war", "risk",
        "disruption", "port", "typhoon", "earthquake", "flood",
        "sanction", "geopolit", "crisis", "alert", "threat", "danger",
        "hurricane", "fire", "pandemic", "inflation", "conflict"
    ],
    "SupplyChainKnowledge": [
        "how do", "what is", "what are", "explain", "define",
        "kpi", "safety stock", "eoq", "abc", "jit", "calculate",
        "recommend", "best practice", "protocol", "guideline",
        "should i", "help me", "tell me"
    ],
    "StatusReport": [
        "status", "report", "overview", "summary", "dashboard",
        "overall", "health", "full report", "everything", "all"
    ]
}


def route_query(query: str) -> str:
    """
    Intelligently routes a query to the best tool.
    Uses keyword scoring — whichever tool scores highest wins.
    """
    query_lower = query.lower()
    scores      = {tool: 0 for tool in TOOL_KEYWORDS}

    for tool, keywords in TOOL_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                scores[tool] += 1

    best_tool = max(scores, key=scores.get)

    # If no keywords matched, default to knowledge base
    if scores[best_tool] == 0:
        best_tool = "SupplyChainKnowledge"

    return best_tool


def run_agent(query: str, session_id: str = "default") -> dict:
    """
    Main agent function — routes query, runs tool, saves to memory.
    This is the core of the Orchestrator Agent.
    """
    print(f"\n🤖 AGENT RECEIVED: '{query}'")

    # Step 1 — Get memory context
    history = get_memory_history(session_id, limit=3)
    context = ""
    if history:
        context = "Recent history:\n"
        for h in history:
            context += f"  - [{h[2]}] {h[1][:50]}...\n"
        print(f"📝 Memory context: {len(history)} past interactions loaded")

    # Step 2 — Route to best tool
    tool_name = route_query(query)
    print(f"🎯 Routing to: {tool_name}")

    # Step 3 — Find and run the tool
    tool_map = {t.name: t for t in AGENT_TOOLS}
    tool     = tool_map[tool_name]

    # For risk tool, pass the query as the headline
    if tool_name == "RiskDetector":
        result = tool.func(query)
    else:
        result = tool.func(query)

    # Step 4 — Save to memory
    save_to_memory(query, tool_name, result, session_id)

    # Step 5 — Save risk alerts separately
    if tool_name == "RiskDetector" and "RISK DETECTED" in result:
        from src.risk_nlp.inference import predict_risk
        r = predict_risk(query)
        if r["is_risk"]:
            save_risk_alert(query, r["risk_category"], r["severity_score"])
            print(f"🚨 Risk alert saved: {r['risk_category']}")

    print(f"✅ Agent response ready from {tool_name}")

    return {
        "query":      query,
        "tool_used":  tool_name,
        "result":     result,
        "session_id": session_id,
        "memory_ctx": context
    }


# ── TEST THE AGENT ───────────────────────────────────────────
if __name__ == "__main__":
    init_memory_db()

    test_queries = [
        "What will be the demand for next week?",
        "Dock workers at LA port are going on strike!",
        "How do I calculate safety stock?",
        "Give me a full status report of supply chain",
        "There is a typhoon hitting Taiwan semiconductor factories",
        "What is the perfect order rate target?"
    ]

    print("=" * 60)
    print("   SMARTCHAIN AGENTIC AI — ORCHESTRATOR TEST")
    print("=" * 60)

    for q in test_queries:
        response = run_agent(q, session_id="test_session")
        print(f"\n📤 RESULT:\n{response['result']}")
        print(f"🔧 Tool used: {response['tool_used']}")
        print("-" * 60)

    # Show memory
    print("\n📝 AGENT MEMORY (last 6 decisions):")
    history = get_memory_history("test_session", limit=6)
    for h in history:
        print(f"  [{h[2]:20s}] {h[1][:50]}")