import os
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from google import genai

# =========================
# CONFIG
# =========================
GENAI_API_KEY = "AIzaSyCDdg-OdX21wI3ypcotDbz447W47XDKud8"  
GEMINI_MODEL = "gemini-2.5-flash-lite"

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Baymax Chat Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# GEMINI CLIENT
# =========================
if not GENAI_API_KEY:
    raise RuntimeError("GENAI_API_KEY not set")

genai_client = genai.Client(api_key=GENAI_API_KEY)

# =========================
# IN-MEMORY CONVERSATION STORE
# session_id -> list of messages
# =========================
conversation_store: Dict[str, List[Dict[str, str]]] = {}

# =========================
# REQUEST / RESPONSE MODELS
# =========================
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

# =========================
# HELPER: BUILD PROMPT
# =========================
def build_prompt(history: List[Dict[str, str]]) -> str:
    """
    Converts chat history into a single prompt for Gemini.
    """
    prompt = """
You are Baymax, a calm, friendly, and caring healthcare assistant.
Provide helpful advice in English.
Be empathetic, natural, and professional.

Conversation so far:
"""
    for msg in history:
        role = msg["role"]
        content = msg["content"]
        prompt += f"{role.upper()}: {content}\n"

    prompt += "ASSISTANT:"
    return prompt

# =========================
# CHAT ENDPOINT
# =========================
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id.strip()
    user_message = request.message.strip()

    if not session_id or not user_message:
        raise HTTPException(status_code=400, detail="session_id and message are required")

    # Initialize session if not exists
    if session_id not in conversation_store:
        conversation_store[session_id] = []

    # Add user message to history
    conversation_store[session_id].append({
        "role": "user",
        "content": user_message
    })

    # Build prompt from history
    prompt = build_prompt(conversation_store[session_id])

    try:
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        assistant_reply = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e}")

    # Add assistant reply to history
    conversation_store[session_id].append({
        "role": "assistant",
        "content": assistant_reply
    })

    return ChatResponse(reply=assistant_reply)

# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def health():
    return {"status": "ok", "message": "Baymax chat backend running"}

# =========================
# RUN
# =========================
if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)
