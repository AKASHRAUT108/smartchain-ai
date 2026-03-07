from transformers import pipeline
import pandas as pd

print("📦 Loading zero-shot classifier...")
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

CANDIDATE_LABELS = [
    "normal supply chain news",
    "port strike or shipping delay",
    "natural disaster disruption",
    "raw material shortage",
    "geopolitical trade disruption"
]

def zero_shot_predict(headline: str) -> dict:
    result = classifier(headline, CANDIDATE_LABELS)
    top_label = result['labels'][0]
    top_score = result['scores'][0]
    return {
        "headline":   headline,
        "prediction": top_label,
        "confidence": round(top_score, 3)
    }

# Test it
headlines = [
    "Longshoremen strike shuts down East Coast ports",
    "New logistics software reduces delivery times by 20%",
    "Earthquake in Japan damages Toyota supply chain",
]

print("\n=== ZERO-SHOT RESULTS ===\n")
for h in headlines:
    result = zero_shot_predict(h)
    print(f"📰 {result['headline']}")
    print(f"   → {result['prediction']} ({result['confidence']:.0%})\n")