from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rasa_client import send_to_rasa
import os

app = FastAPI()

# Allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    response = send_to_rasa(request.session_id, request.message)
    return response

# In production (Hugging Face), static files are in /app/static
# In development, they might be in ../static relative to this file
static_dir = "/app/static"
if not os.path.exists(static_dir):
    static_dir = "../static"

if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
else:
    print(f"Warning: static directory {static_dir} not found")
