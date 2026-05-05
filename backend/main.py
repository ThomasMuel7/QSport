from __future__ import annotations
from typing import Any
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from agent import get_agent
from rag import get_context

load_dotenv()

app = FastAPI(title="QSport Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[dict[str, Any]] = Field(default_factory=list)

class ChatResponse(BaseModel):
    response: str
    sources: list[dict[str, Any]]

class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    league: str
    date: str

class PredictResponse(BaseModel):
    winner: str
    confidence: float
    explanation: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    context = get_context(payload.message)
    agent = get_agent()

    context_str = ""
    if context:
        lines = [c.get("content", "") for c in context if not c.get("metadata", {}).get("fallback")]
        if lines:
            context_str = "\nContexte disponible:\n" + "\n".join(lines)

    full_message = payload.message + context_str
    response = agent.run(full_message)
    return ChatResponse(response=response, sources=context)

@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    return PredictResponse(
        winner=payload.home_team,
        confidence=0.6,
        explanation=f"Prediction en cours pour {payload.home_team} vs {payload.away_team}."
    )
