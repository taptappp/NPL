import os
from fastapi import FastAPI
from pydantic import BaseModel

from core.core_logic import sentiment_service

app = FastAPI(
    title="PhoBERT Sentiment API",
    version="1.0"
)

class TextRequest(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "ok", "message": "API is running"}

@app.post("/predict")
def predict(req: TextRequest):
    return sentiment_service.predict_sentiment(req.text)

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
