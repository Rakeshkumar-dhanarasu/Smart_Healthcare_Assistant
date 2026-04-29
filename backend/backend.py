import os
from typing import Dict, List

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import shutil
import uuid
import os

# ✅ Your Hybrid RAG Function
from rag_function import get_answer
from image_classification import ocr_rag_pipeline


# ---------------- APP INIT ----------------
app = FastAPI(title="Baymax Chat Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- MODELS ----------------
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


# ---------------- OPTIONAL MEMORY (can extend later) ----------------
conversation_store: Dict[str, List[Dict[str, str]]] = {}

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/image-upload")
async def image_upload(
    session_id: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Receives an image file, passes it through ocr_rag_pipeline,
    and returns Baymax's reply.
    """
    try:
        # Save uploaded image to disk temporarily
        unique_filename = f"{uuid.uuid4()}_{image.filename}"
        image_path = os.path.join(UPLOAD_DIR, unique_filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Process the image with OCR + RAG pipeline
        # Expected to return a string reply
        reply = ocr_rag_pipeline(image_path)

        # Optional: Delete the file after processing
        os.remove(image_path)

        return JSONResponse({"reply": reply})

    except Exception as e:
        return JSONResponse({"reply": f"Error processing image: {str(e)}"}, status_code=500)


# ---------------- CHAT ENDPOINT ----------------
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    session_id = request.session_id.strip()
    user_message = request.message.strip()

    if not session_id or not user_message:
        raise HTTPException(status_code=400, detail="session_id and message are required")

    try:
        result = get_answer(user_message)
        assistant_reply = result["answer"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")

    return ChatResponse(reply=assistant_reply)


# ---------------- CARE REMEDY ENDPOINT ----------------
@app.post("/care-remedy", response_model=CareRemedyResponse)
async def care_remedy(request: CareRemedyRequest):

    symptom = request.symptom.strip()

    if not symptom:
        raise HTTPException(status_code=400, detail="Symptom is required")

    try:
        result = get_answer(symptom)
        remedy = result["answer"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")

    return CareRemedyResponse(remedy=remedy)


# ---------------- HEALTH CHECK ----------------
@app.get("/")
def health():
    return {"status": "ok", "message": "Baymax backend running 🚀"}


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=False)