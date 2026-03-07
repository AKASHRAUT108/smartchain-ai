import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import json

load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# ─── SEARCH QUERIES ────────────────────────────────────────
# These keywords fetch supply chain relevant news
SUPPLY_CHAIN_QUERIES = [
    "supply chain disruption",
    "port strike shipping delay",
    "raw material shortage",
    "logistics crisis",
    "semiconductor shortage",
    "freight delay",
    "factory shutdown",
    "trade war tariff"
]

def fetch_supply_chain_news(days_back=7, max_articles=100):
    """
    Fetch recent supply chain news from NewsAPI.
    Returns a DataFrame with title, description, source, date.
    """
    all_articles = []
    from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')

    print(f"📰 Fetching news from last {days_back} days...")

    for query in SUPPLY_CHAIN_QUERIES:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q":        query,
            "from":     from_date,
            "sortBy":   "relevancy",
            "language": "en",
            "pageSize": 10,          # 10 articles per query
            "apiKey":   NEWSAPI_KEY
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if data.get("status") == "ok":
                articles = data.get("articles", [])
                for article in articles:
                    all_articles.append({
                        "title":       article.get("title", ""),
                        "description": article.get("description", ""),
                        "source":      article.get("source", {}).get("name", ""),
                        "published":   article.get("publishedAt", ""),
                        "url":         article.get("url", ""),
                        "query":       query
                    })
                print(f"   ✅ '{query}' → {len(articles)} articles")
            else:
                print(f"   ⚠️  '{query}' → API error: {data.get('message')}")

        except Exception as e:
            print(f"   ❌ '{query}' → {e}")

    # Remove duplicates by title
    df = pd.DataFrame(all_articles)
    df = df.drop_duplicates(subset='title').reset_index(drop=True)
    df = df[df['title'].notna() & (df['title'] != '')]

    print(f"\n📦 Total unique articles fetched: {len(df)}")
    return df


def create_training_dataset():
    """
    Create a labeled dataset for BERT fine-tuning.
    Labels: 0=Normal, 1=Port/Strike, 2=Natural Disaster,
            3=Raw Material, 4=Geopolitical
    """
    # Hand-labeled training examples
    # In a real project you'd label 500-1000 examples
    # We create a solid starter set of 60 examples here

    data = [
        # ── NORMAL (label=0) ──────────────────────────────────
        ("Supply chain management improves efficiency across retail sector", 0),
        ("Companies invest in digital transformation for logistics", 0),
        ("Global trade volumes remain stable in Q3", 0),
        ("Warehouse automation reduces operational costs by 15%", 0),
        ("New shipping routes open between Asia and Europe", 0),
        ("Logistics companies report steady growth this quarter", 0),
        ("Inventory management systems upgraded across major retailers", 0),
        ("Supply chain visibility tools gain adoption in manufacturing", 0),
        ("E-commerce fulfillment centers expand capacity nationwide", 0),
        ("Freight rates stabilize after months of volatility", 0),
        ("New trade agreements benefit cross-border commerce", 0),
        ("Supply chain software market grows 12% annually", 0),

        # ── PORT / STRIKE (label=1) ───────────────────────────
        ("Dock workers strike at major US port causing shipping delays", 1),
        ("Port congestion at Los Angeles leads to weeks-long cargo backup", 1),
        ("Longshoremen union votes for strike amid wage disputes", 1),
        ("Rotterdam port operations halted due to worker strike", 1),
        ("Container ship backlog grows as port workers walk out", 1),
        ("Strike at Singapore port disrupts Asian supply chains", 1),
        ("West Coast port labor dispute delays holiday merchandise", 1),
        ("Truck drivers strike causes port congestion nationwide", 1),
        ("Port of Shanghai closes temporarily due to labor action", 1),
        ("Major ports across Europe face strike threats this week", 1),
        ("Shipping containers pile up as dock workers extend strike", 1),
        ("Port workers demand 30% pay rise, halt operations", 1),

        # ── NATURAL DISASTER (label=2) ────────────────────────
        ("Typhoon shuts down factories across Southeast Asia", 2),
        ("Flooding in Thailand disrupts automotive supply chains globally", 2),
        ("Hurricane damages Gulf Coast chemical plants and refineries", 2),
        ("Earthquake in Japan halts semiconductor manufacturing plants", 2),
        ("Severe drought reduces crop yields causing food supply shortage", 2),
        ("Wildfire destroys warehouse facilities in California", 2),
        ("Tsunami warning disrupts coastal manufacturing operations", 2),
        ("Record snowstorm shuts down transportation networks across midwest", 2),
        ("Volcano eruption in Iceland disrupts European air freight", 2),
        ("Monsoon season causes widespread flooding in Bangladesh factories", 2),
        ("Forest fires force evacuation of industrial zones in Canada", 2),
        ("Extreme heatwave shuts down European factories amid power shortage", 2),

        # ── RAW MATERIAL SHORTAGE (label=3) ───────────────────
        ("Global semiconductor chip shortage cripples auto production", 3),
        ("Lithium supply crunch threatens electric vehicle battery output", 3),
        ("Steel prices surge 40% amid global raw material shortage", 3),
        ("Rare earth metal shortage disrupts electronics manufacturing", 3),
        ("Cotton shortage drives apparel prices to record highs", 3),
        ("Aluminum shortage halts production lines at major manufacturers", 3),
        ("Natural gas shortage forces European factories to reduce output", 3),
        ("Copper supply deficit threatens renewable energy projects", 3),
        ("Wheat shortage causes global food supply chain disruptions", 3),
        ("Resin shortage impacts plastic packaging supply globally", 3),
        ("Silicon shortage extends semiconductor lead times to 52 weeks", 3),
        ("Cobalt supply squeeze threatens battery manufacturing capacity", 3),

        # ── GEOPOLITICAL (label=4) ────────────────────────────
        ("US imposes new tariffs on Chinese goods disrupting imports", 4),
        ("Russia Ukraine war impacts global wheat and fertilizer exports", 4),
        ("Trade war escalates as countries impose retaliatory tariffs", 4),
        ("Sanctions on Iran restrict oil supply affecting energy costs", 4),
        ("Brexit causes delays at UK customs disrupting European supply chains", 4),
        ("US China tensions lead to export controls on semiconductors", 4),
        ("Middle East conflict disrupts oil tanker routes through Strait", 4),
        ("New import regulations create bottlenecks at border crossings", 4),
        ("Political instability in South America disrupts mining operations", 4),
        ("Trade embargo on Russia impacts global commodity markets", 4),
        ("North Korea missile tests disrupt shipping lanes in Pacific", 4),
        ("EU carbon border tax reshapes global manufacturing supply chains", 4),
    ]

    df = pd.DataFrame(data, columns=['text', 'label'])

    label_map = {
        0: "Normal",
        1: "Port_Strike",
        2: "Natural_Disaster",
        3: "Raw_Material_Shortage",
        4: "Geopolitical"
    }
    df['label_name'] = df['label'].map(label_map)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/risk_training_data.csv", index=False)

    print("✅ Training dataset created!")
    print(df['label_name'].value_counts())
    return df


if __name__ == "__main__":
    # Test news fetching
    print("=== TESTING NEWS FETCHER ===\n")
    news_df = fetch_supply_chain_news(days_back=3, max_articles=50)
    print(news_df[['title', 'source', 'published']].head(10))

    # Save fetched news
    news_df.to_csv("data/processed/live_news.csv", index=False)
    print("\n✅ Live news saved to data/processed/live_news.csv")

    # Create training data
    print("\n=== CREATING TRAINING DATASET ===\n")
    train_df = create_training_dataset()