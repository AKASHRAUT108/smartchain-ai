from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(
    title="SmartChain AI",
    description="Intelligent Supply Chain Risk Intelligence & Demand Forecasting",
    version="1.0.0"
)

# Allow dashboard to talk to API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "project": "SmartChain AI",
        "modules": ["forecasting", "risk_nlp", "rag"]
    }

@app.get("/")
def root():
    return {"message": "Welcome to SmartChain AI API 🚀"}

# We will add more routes here in later steps
# POST /forecast
# GET  /risk-feed
# POST /chat

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)