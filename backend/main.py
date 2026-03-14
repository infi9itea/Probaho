import re
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rasa_client import send_to_rasa
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure consistent language detection
DetectorFactory.seed = 0

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

def is_bangla(text: str) -> bool:
    """Detect if text contains Bangla script or common Banglish keywords"""
    if not text:
        return False
    # Check for Bangla Unicode range
    for char in text:
        if '\u0980' <= char <= '\u09FF':
            return True
    # Expanded Banglish words - removed ambiguous English words (admission, deadline, fee)
    banglish_words = {
        'ami', 'tumi', 'koto', 'borti', 'hote', 'chai', 'kivabe', 'jonno', 'khoroch',
        'ache', 'nai', 'salam', 'kemon', 'apni', 'ki', 'thikana',
        'kobe', 'kar', 'sathe', 'jogajog', 'korte', 'parbo', 'bolun', 'bolte', 'paren',
        'kothay', 'kakhon', 'khola', 'pabo', 'dekha', 'janan'
    }
    words = set(re.findall(r'\w+', text.lower()))
    if words.intersection(banglish_words):
        return True
    return False

def detect_language(text: str) -> str:
    """Identify the language of the text (bn, en, etc.)"""
    if not text or not text.strip():
        return 'en'

    if is_bangla(text):
        return 'bn'

    try:
        return detect(text)
    except Exception:
        return 'en'

def translate_if_needed(text: str, target_lang: str):
    """Translates text ONLY if it's English and the target is different."""
    if not text or not text.strip() or target_lang == 'en':
        return text

    # Detect language of the bot's response
    # If it's already in a non-English language (like Bangla), don't translate it again
    bot_lang = detect_language(text)
    if bot_lang != 'en':
        return text

    try:
        translated = GoogleTranslator(source='en', target=target_lang).translate(text)
        logger.info(f"Translated output from EN to {target_lang}: {translated[:50]}...")
        return translated
    except Exception as e:
        logger.error(f"Translation from EN to {target_lang} failed: {e}")
        return text

@app.post("/chat")
async def chat(request: ChatRequest):
    # 1. Detect language of the query
    original_lang = detect_language(request.message)

    # 2. Send the ORIGINAL message to Rasa (let Rasa/Action/RAG handle language natively)
    responses = send_to_rasa(request.session_id, request.message)

    # 3. If the original message was not English, translate Rasa's English responses back
    if original_lang != 'en' and isinstance(responses, list):
        for resp in responses:
            if 'text' in resp:
                resp['text'] = translate_if_needed(resp['text'], original_lang)

    return responses
