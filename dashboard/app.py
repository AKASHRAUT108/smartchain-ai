import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta

# ─── CONFIG ────────────────────────────────────────────────
API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title = "SmartChain AI",
    page_icon  = "🏭",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: 800;
        background: linear-gradient(90deg, #1F4E79, #2E75B6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background: #f0f4ff; border-radius: 12px;
        padding: 1rem; border-left: 4px solid #2E75B6;
    }
    .risk-high   { color: #C00000; font-weight: bold; }
    .risk-medium { color: #FF6600; font-weight: bold; }
    .risk-low    { color: #375623; font-weight: bold; }
    .stButton>button {
        background: #2E75B6; color: white;
        border-radius: 8px; border: none;
        padding: 0.5rem 1.5rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/supply-chain.png", width=80)
    st.markdown("## 🏭 SmartChain AI")
    st.markdown("*Intelligent Supply Chain Intelligence*")
    st.divider()

    # API health check
    try:
        resp = requests.get(f"{API_URL}/health", timeout=3)
        if resp.status_code == 200:
            st.success("🟢 API Connected")
        else:
            st.error("🔴 API Error")
    except:
        st.error("🔴 API Offline\nStart with: uvicorn src.api.main:app")

    st.divider()
    st.markdown("### 📌 Navigation")
    page = st.radio(
    "Go to",
    ["📈 Demand Forecast",
     "⚠️ Risk Analyzer",
     "🤖 AI Chatbot",
     "📊 Dashboard Overview",
     "🧠 Agent Console"],        # ← ADD THIS
    label_visibility="collapsed"
    )
    st.divider()
    st.markdown("**Models Active:**")
    st.markdown("✅ LSTM Forecasting")
    st.markdown("✅ BERT Risk NLP")
    st.markdown("✅ RAG Chatbot")

# ══════════════════════════════════════════════════
# PAGE 1 — DEMAND FORECAST
# ══════════════════════════════════════════════════
if page == "📈 Demand Forecast":
    st.markdown('<p class="main-header">📈 Demand Forecasting</p>',
                unsafe_allow_html=True)
    st.markdown("LSTM model predicting future product demand from historical patterns.")
    st.divider()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### ⚙️ Forecast Settings")
        days     = st.slider("Forecast horizon (days)", 1, 30, 7)
        category = st.selectbox(
            "Product Category",
            ["all", "electronics", "clothing", "furniture", "sports"]
        )
        if st.button("🚀 Generate Forecast"):
            with st.spinner("Running LSTM model..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/forecast",
                        json={"days": days, "product_category": category},
                        timeout=30
                    )
                    data = resp.json()
                    st.session_state["forecast_data"] = data
                    st.success("✅ Forecast complete!")
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        if "forecast_data" in st.session_state:
            data  = st.session_state["forecast_data"]
            preds = data["predictions"]
            dates = [
                (datetime.now() + timedelta(days=i)).strftime("%b %d")
                for i in range(len(preds))
            ]

            # Plotly line chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates, y=preds,
                mode="lines+markers",
                name="Predicted Demand",
                line=dict(color="#2E75B6", width=3),
                marker=dict(size=8),
                fill="tozeroy",
                fillcolor="rgba(46,117,182,0.1)"
            ))
            fig.update_layout(
                title=f"Demand Forecast — Next {days} Days",
                xaxis_title="Date",
                yaxis_title="Units",
                height=350,
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            # Metrics row
            m1, m2, m3 = st.columns(3)
            m1.metric("📦 Avg Daily",  f"{sum(preds)/len(preds):.0f} units")
            m2.metric("📈 Peak Day",   f"{max(preds):.0f} units")
            m3.metric("📉 Trend",
                      "⬆️ Increasing" if data["trend"] == "increasing"
                      else "⬇️ Decreasing")
        else:
            st.info("👈 Set parameters and click **Generate Forecast**")

# ══════════════════════════════════════════════════
# PAGE 2 — RISK ANALYZER
# ══════════════════════════════════════════════════
elif page == "⚠️ Risk Analyzer":
    st.markdown('<p class="main-header">⚠️ Risk Analyzer</p>',
                unsafe_allow_html=True)
    st.markdown("BERT model classifying supply chain risk from news headlines.")
    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📰 Analyze a Headline")
        headline = st.text_area(
            "Enter a news headline:",
            placeholder="e.g. Dock workers at LA port go on strike...",
            height=100
        )
        if st.button("🔍 Analyze Risk"):
            if headline.strip():
                with st.spinner("Running BERT classifier..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/risk",
                            json={"headline": headline},
                            timeout=15
                        )
                        result = resp.json()
                        st.session_state["risk_result"] = result
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a headline")

    with col2:
        if "risk_result" in st.session_state:
            r = st.session_state["risk_result"]

            # Risk badge
            color_map = {
                "Normal":                "🟢",
                "Port_Strike":           "🔴",
                "Natural_Disaster":      "🔴",
                "Raw_Material_Shortage": "🟠",
                "Geopolitical":          "🟠"
            }
            badge = color_map.get(r["risk_category"], "⚪")

            st.markdown(f"### {badge} {r['risk_category'].replace('_', ' ')}")
            st.metric("Confidence",     f"{r['confidence']:.1%}")
            st.metric("Severity Score", f"{r['severity_score']} / 10")

            if r["is_risk"]:
                st.error("⚠️ Supply chain risk detected!")
            else:
                st.success("✅ No significant risk detected")

            # Probability bar chart
            probs = r.get("all_probabilities", {})
            if probs:
                fig = px.bar(
                    x=list(probs.keys()),
                    y=list(probs.values()),
                    color=list(probs.values()),
                    color_continuous_scale="RdYlGn_r",
                    title="Risk Category Probabilities"
                )
                fig.update_layout(height=280, showlegend=False,
                                  coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)

    # Sample headlines to test
    st.divider()
    st.markdown("### 🧪 Quick Test Headlines")
    samples = [
        "Dock workers at major US port go on strike causing shipping delays",
        "Supply chain software company reports record quarterly revenue growth",
        "Typhoon hits Taiwan disrupting semiconductor manufacturing plants",
        "US imposes 25% tariffs on Chinese electronics imports",
        "Global lithium shortage threatens electric vehicle battery supply"
    ]
    for s in samples:
        if st.button(f"📋 {s[:60]}...", key=s):
            st.session_state["prefill"] = s
            st.rerun()

# ══════════════════════════════════════════════════
# PAGE 3 — AI CHATBOT
# ══════════════════════════════════════════════════
elif page == "🤖 AI Chatbot":
    st.markdown('<p class="main-header">🤖 SmartChain AI Chatbot</p>',
                unsafe_allow_html=True)
    st.markdown("RAG-powered assistant answering supply chain questions.")
    st.divider()

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "confidence" in msg:
                st.caption(f"📊 Confidence: {msg['confidence']:.1%} | "
                           f"📎 {msg['sources']} sources retrieved")

    # Chat input
    if user_input := st.chat_input("Ask about supply chain risks, inventory, KPIs..."):
        # Show user message
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.chat_history.append(
            {"role": "user", "content": user_input}
        )

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/chat",
                        json={"message": user_input},
                        timeout=30
                    )
                    result = resp.json()
                    answer = result["answer"]
                    conf   = result.get("confidence", 0)
                    srcs   = len(result.get("sources", []))
                    st.write(answer)
                    st.caption(
                        f"📊 Confidence: {conf:.1%} | 📎 {srcs} sources retrieved"
                    )
                    st.session_state.chat_history.append({
                        "role":       "assistant",
                        "content":    answer,
                        "confidence": conf,
                        "sources":    srcs
                    })
                except Exception as e:
                    st.error(f"API Error: {e}")

    # Suggested questions
    st.divider()
    st.markdown("### 💡 Try asking:")
    q_col1, q_col2 = st.columns(2)
    with q_col1:
        st.markdown("- How do I calculate safety stock?")
        st.markdown("- What is the perfect order rate target?")
        st.markdown("- How should I respond to a port strike?")
    with q_col2:
        st.markdown("- What is EOQ formula?")
        st.markdown("- How much does air freight cost?")
        st.markdown("- What are A-items in ABC analysis?")

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ══════════════════════════════════════════════════
# PAGE 4 — DASHBOARD OVERVIEW
# ══════════════════════════════════════════════════
elif page == "📊 Dashboard Overview":
    st.markdown('<p class="main-header">📊 Dashboard Overview</p>',
                unsafe_allow_html=True)
    st.markdown("System status and quick metrics at a glance.")
    st.divider()

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🤖 Models Active",     "3 / 3",  "All healthy")
    c2.metric("📰 Risk Categories",   "5",       "Classified")
    c3.metric("📚 KB Documents",      "2",       "Indexed")
    c4.metric("🔗 API Endpoints",     "5",       "Live")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧠 Model Architecture")
        arch_data = {
            "Module":    ["Demand Forecast", "Risk NLP",    "RAG Chatbot",  "Vector Store"],
            "Model":     ["LSTM + 1D-CNN",   "DistilBERT",  "RoBERTa QA",   "FAISS"],
            "Status":    ["✅ Active",        "✅ Active",    "✅ Active",     "✅ Active"],
            "Task":      ["Regression",       "Classification","Extractive QA","Similarity"]
        }
        st.dataframe(
            pd.DataFrame(arch_data),
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.markdown("### ⚠️ Risk Category Distribution")
        risk_cats = ["Normal", "Port Strike", "Natural Disaster",
                     "Raw Material", "Geopolitical"]
        # Sample distribution for demo
        counts = [12, 12, 12, 12, 12]
        fig = px.pie(
            values=counts, names=risk_cats,
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4
        )
        fig.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### 🚀 Quick Actions")
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("📈 Run 7-Day Forecast"):
            with st.spinner("Forecasting..."):
                try:
                    r = requests.post(
                        f"{API_URL}/forecast",
                        json={"days": 7}, timeout=30
                    )
                    d = r.json()
                    st.success(
                        f"Trend: {d['trend']} | "
                        f"Avg: {sum(d['predictions'])/len(d['predictions']):.0f} units/day"
                    )
                except Exception as e:
                    st.error(str(e))
    with a2:
        if st.button("⚠️ Check API Health"):
            try:
                r = requests.get(f"{API_URL}/health", timeout=3)
                st.success(f"✅ {r.json()}")
            except:
                st.error("❌ API offline")
    with a3:
        if st.button("📰 Get Risk Feed"):
            try:
                r = requests.get(f"{API_URL}/risk-feed", timeout=10)
                d = r.json()
                st.info(
                    f"📰 {d.get('total_articles', 0)} articles | "
                    f"🚨 {d.get('risk_alerts', 0)} alerts"
                )
            except Exception as e:
                st.error(str(e))

# ══════════════════════════════════════════════════
elif page == "🧠 Agent Console":
    st.markdown('<p class="main-header">🧠 Agent Console</p>',
                unsafe_allow_html=True)
    st.markdown("Orchestrator Agent — routes your query to the right AI automatically.")
    st.divider()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 💬 Talk to the Agent")
        session_id = st.text_input("Session ID", value="my_session")
        query      = st.text_area(
            "Ask the agent anything:",
            placeholder="e.g. There is a port strike in LA!\nor: What will demand be next week?\nor: Give me a full status report",
            height=120
        )

        if st.button("🚀 Run Agent"):
            if query.strip():
                with st.spinner("🤖 Agent thinking..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/agent/run",
                            json={"query": query, "session_id": session_id},
                            timeout=60
                        )
                        data = resp.json()
                        st.session_state["agent_result"] = data
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a query")

        # Quick test buttons
        st.markdown("### ⚡ Quick Tests")
        quick = [
            "What will demand be next week?",
            "Port strike at LA causing delays!",
            "How do I calculate safety stock?",
            "Give me a full status report"
        ]
        for q in quick:
            if st.button(q, key=f"quick_{q}"):
                with st.spinner("🤖 Agent thinking..."):
                    try:
                        resp = requests.post(
                            f"{API_URL}/agent/run",
                            json={"query": q, "session_id": session_id},
                            timeout=60
                        )
                        st.session_state["agent_result"] = resp.json()
                    except Exception as e:
                        st.error(str(e))

    with col2:
        if "agent_result" in st.session_state:
            data = st.session_state["agent_result"]

            # Tool badge
            tool_colors = {
                "DemandForecaster":     "🟦",
                "RiskDetector":         "🟥",
                "SupplyChainKnowledge": "🟩",
                "StatusReport":         "🟨"
            }
            badge = tool_colors.get(data["tool_used"], "⬜")
            st.markdown(f"### {badge} Tool Used: `{data['tool_used']}`")
            st.text_area("Agent Response:", value=data["result"], height=300)

            if data.get("memory_ctx"):
                st.caption(f"📝 Memory: {data['memory_ctx']}")

    # Risk Alerts Panel
    st.divider()
    st.markdown("### 🚨 Autonomous Risk Alerts")
    try:
        resp   = requests.get(f"{API_URL}/agent/alerts", timeout=10)
        alerts = resp.json()
        if alerts["total_alerts"] > 0:
            for a in alerts["alerts"][:5]:
                severity = a["severity"]
                color    = "🔴" if severity >= 7 else "🟠" if severity >= 4 else "🟡"
                st.markdown(
                    f"{color} **[{a['risk_cat']}]** (severity: {severity}/10)  \n"
                    f"_{a['headline'][:80]}_  \n"
                    f"<small>{a['timestamp'][:19]}</small>",
                    unsafe_allow_html=True
                )
        else:
            st.info("No risk alerts yet. Run the monitor agent to start scanning.")
    except:
        st.warning("Start the API to see alerts")

    # Memory Panel
    st.divider()
    st.markdown("### 📝 Agent Memory")
    try:
        resp    = requests.get(f"{API_URL}/agent/memory/{session_id}", timeout=10)
        mem     = resp.json()
        if mem["history"]:
            mem_df = pd.DataFrame(mem["history"])
            st.dataframe(mem_df[["timestamp","query","tool_used"]],
                        use_container_width=True, hide_index=True)
        else:
            st.info("No memory yet for this session.")
    except:
        st.warning("Start the API to see memory")