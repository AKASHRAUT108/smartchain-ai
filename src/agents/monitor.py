import sys, os, time, schedule
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.agents.orchestrator import run_agent, save_risk_alert, init_memory_db
from src.risk_nlp.inference import predict_risk

# Sample headlines to monitor (replace with NewsAPI when available)
MONITOR_HEADLINES = [
    "Port workers threaten strike at East Coast terminals",
    "Semiconductor supply remains stable this quarter",
    "Floods in Bangladesh disrupt garment factory output",
    "New trade agreement reduces tariffs on electronics",
    "Typhoon warning issued for Taiwan manufacturing region",
    "Global freight rates decrease for third consecutive month"
]

def autonomous_risk_scan():
    """
    Runs every N minutes automatically.
    Scans headlines, detects risks, saves alerts.
    No human trigger needed — fully autonomous.
    """
    print(f"\n🤖 [AUTO-SCAN] Running autonomous risk scan...")
    alerts_found = 0

    for headline in MONITOR_HEADLINES:
        result = predict_risk(headline)
        if result["is_risk"]:
            save_risk_alert(
                headline,
                result["risk_category"],
                result["severity_score"]
            )
            alerts_found += 1
            print(f"   🚨 Alert: [{result['risk_category']}] {headline[:50]}...")

    print(f"   ✅ Scan complete — {alerts_found} new alerts saved\n")


def run_monitor(interval_minutes: int = 1):
    """
    Start the autonomous monitoring agent.
    Runs risk scan every N minutes automatically.
    """
    init_memory_db()
    print(f"🚀 Autonomous Monitor Agent started")
    print(f"   Scanning every {interval_minutes} minute(s)")
    print(f"   Press Ctrl+C to stop\n")

    # Run immediately on start
    autonomous_risk_scan()

    # Then schedule recurring scans
    schedule.every(interval_minutes).minutes.do(autonomous_risk_scan)

    while True:
        schedule.run_pending()
        time.sleep(10)


if __name__ == "__main__":
    run_monitor(interval_minutes=1)