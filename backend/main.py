from fastapi import FastAPI
from pydantic import BaseModel
from rasa_client import send_to_rasa

app = FastAPI()

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    response = send_to_rasa(request.session_id, request.message)
    return response
