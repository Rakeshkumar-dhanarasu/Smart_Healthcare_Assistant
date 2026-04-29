import os
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from google import genai


GENAI_API_KEY = "AIzaSyBb9_QaTgKSoj8d9Ymz1xqhmil48lTVrNg"   #AIzaSyCDdg-OdX21wI3ypcotDbz447W47XDKud8
GEMINI_MODEL = "gemini-2.5-flash-lite"

app = FastAPI(title="Baymax Chat Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not GENAI_API_KEY:
    raise RuntimeError("GENAI_API_KEY not set")

genai_client = genai.Client(api_key=GENAI_API_KEY)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

# Load FAISS nutrients database
nutrients_db = FAISS.load_local(
    "vector_db/nutrients_db",   # <-- your saved folder
    embeddings,
    allow_dangerous_deserialization=True
)

def get_nutrient_context(query: str) -> str:
    docs = nutrient_retriever.get_relevant_documents(query)

    if not docs:
        return ""

    context = "\n\n".join(doc.page_content for doc in docs)

    return f"""
    Relevant Nutritional Knowledge:
    {context}
    """

nutrient_retriever = nutrients_db.as_retriever(search_kwargs={"k": 3})

conversation_store: Dict[str, List[Dict[str, str]]] = {}

class CareRemedyRequest(BaseModel):
    profile: Dict[str, str]
    symptom: str

class CareRemedyResponse(BaseModel):
    remedy: str

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str


def build_prompt(history: List[Dict[str, str]]) -> str:

    latest_user_message = history[-1]["content"]

    nutrient_context = get_nutrient_context(latest_user_message)

    prompt = f"""
    You are Baymax, an intelligent healthcare assistant.

    Use the provided nutritional knowledge to give accurate, safe, and helpful advice.

    {nutrient_context}

    Conversation so far:
    """

    for msg in history:
        role = msg["role"]
        content = msg["content"]
        prompt += f"{role.upper()}: {content}\n"

    prompt += "ASSISTANT:"

    return prompt


@app.post("/care-remedy", response_model=CareRemedyResponse)
async def care_remedy(request: CareRemedyRequest):

    profile = request.profile
    symptom = request.symptom

    nutrient_context = get_nutrient_context(symptom)

    prompt = f"""
    You are Baymax, a gentle healthcare assistant.

    {nutrient_context}

    Patient details:
    Name: {profile.get('name')}
    Age: {profile.get('age')}
    Gender: {profile.get('gender')}
    Allergies: {profile.get('allergies')}
    Region: {profile.get('region')}
    Symptoms: {symptom}

    TASK:
    Return ONLY ONE short natural remedy sentence.
    """

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id.strip()
    user_message = request.message.strip()

    if not session_id or not user_message:
        raise HTTPException(status_code=400, detail="session_id and message are required")

    
    if session_id not in conversation_store:
        conversation_store[session_id] = []

    
    conversation_store[session_id].append({
        "role": "user",
        "content": user_message
    })

    
    prompt = build_prompt(conversation_store[session_id])

    try:
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )
        assistant_reply = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e}")

    
    conversation_store[session_id].append({
        "role": "assistant",
        "content": assistant_reply
    })

    return ChatResponse(reply=assistant_reply)


@app.get("/")
def health():
    return {"status": "ok", "message": "Baymax chat backend running"}

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)
