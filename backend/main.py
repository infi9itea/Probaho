import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rasa_client import send_to_rasa
from translation_service import translate_to_english, translate_to_bangla, is_bangla

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    message = request.message
    needs_translation = False

    # 1. Detect if the message is in Bangla
    if is_bangla(message):
        logger.info("Bangla detected. Translating to English.")
        needs_translation = True
        message = translate_to_english(message)
        logger.info(f"Translated message: {message}")

    # 2. Send the message to Rasa
    responses = send_to_rasa(request.session_id, message)

    # 3. If translation was needed, translate Rasa's response back to Bangla
    if needs_translation and isinstance(responses, list):
        for resp in responses:
            if 'text' in resp:
                logger.info(f"Translating Rasa response back to Bangla: {resp['text']}")
                resp['text'] = translate_to_bangla(resp['text'])

    return responses
