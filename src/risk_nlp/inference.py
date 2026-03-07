import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_PATH = "models/bert_risk_classifier/final/"

LABEL_NAMES = [
    "Normal",
    "Port_Strike",
    "Natural_Disaster",
    "Raw_Material_Shortage",
    "Geopolitical"
]

SEVERITY_MAP = {
    "Normal":                 0,
    "Port_Strike":            7,
    "Natural_Disaster":       9,
    "Raw_Material_Shortage":  6,
    "Geopolitical":           8
}

# ─── LOAD MODEL ────────────────────────────────────────────
print("📦 Loading BERT risk classifier...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model     = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()
print("✅ Model ready\n")

def predict_risk(text: str) -> dict:
    """
    Predict risk category and severity for a news headline.
    Returns dict with label, confidence, severity score.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs   = torch.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs).item()
        confidence = probs[0][pred_id].item()

    label    = LABEL_NAMES[pred_id]
    severity = SEVERITY_MAP[label]

    # Scale severity by confidence
    adjusted_severity = round(severity * confidence, 1)

    return {
        "text":              text,
        "risk_category":     label,
        "confidence":        round(confidence, 3),
        "severity_score":    adjusted_severity,
        "is_risk":           label != "Normal",
        "all_probabilities": {
            name: round(probs[0][i].item(), 3)
            for i, name in enumerate(LABEL_NAMES)
        }
    }


def analyze_news_feed(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run risk analysis on a full news DataFrame.
    Returns DataFrame with risk predictions added.
    """
    results = []
    print(f"🔍 Analyzing {len(news_df)} articles...")

    for _, row in news_df.iterrows():
        text = str(row.get('title', '')) + " " + str(row.get('description', ''))
        result = predict_risk(text.strip())
        results.append({
            "title":          row.get('title', ''),
            "source":         row.get('source', ''),
            "published":      row.get('published', ''),
            "risk_category":  result['risk_category'],
            "confidence":     result['confidence'],
            "severity_score": result['severity_score'],
            "is_risk":        result['is_risk'],
            "url":            row.get('url', '')
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('severity_score', ascending=False)
    return results_df


if __name__ == "__main__":
    # ── Test with sample headlines ──────────────────────────
    test_headlines = [
        "Dock workers at LA port go on strike causing major shipping delays",
        "Supply chain software company reports record quarterly growth",
        "Typhoon hits Taiwan disrupting TSMC semiconductor production",
        "US imposes 25% tariffs on all Chinese electronics imports",
        "Lithium shortage threatens EV battery supply chains globally",
        "New warehouse opens creating 500 jobs in Ohio logistics hub"
    ]

    print("=== RISK PREDICTION DEMO ===\n")
    for headline in test_headlines:
        result = predict_risk(headline)
        status = "🚨" if result['is_risk'] else "✅"
        print(f"{status} [{result['risk_category']}] "
              f"(conf: {result['confidence']:.0%}, "
              f"severity: {result['severity_score']}/10)")
        print(f"   {headline}\n")

    # ── Analyze live news if available ─────────────────────
    news_path = "data/processed/live_news.csv"
    if os.path.exists(news_path):
        print("\n=== ANALYZING LIVE NEWS FEED ===\n")
        news_df     = pd.read_csv(news_path)
        analyzed_df = analyze_news_feed(news_df)

        print("\n🔴 TOP RISK ALERTS:")
        print(analyzed_df[analyzed_df['is_risk'] == True]
              [['title', 'risk_category', 'severity_score']]
              .head(5).to_string(index=False))

        analyzed_df.to_csv(
            "data/processed/analyzed_news.csv", index=False
        )
        print("\n✅ Full analysis saved to data/processed/analyzed_news.csv")