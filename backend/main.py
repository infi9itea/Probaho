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
    # Common Banglish words
    banglish_words = {'ami', 'tumi', 'koto', 'borti', 'hote', 'chai', 'kivabe', 'jonno', 'khoroch', 'ache', 'nai', 'salam', 'kemon'}
    words = set(re.findall(r'\w+', text.lower()))
    if words.intersection(banglish_words):
        return True
    return False

def translate_to_en(text: str):
    """Translates input text to English, detecting Bangla/Banglish specifically."""
    if not text or not text.strip():
        return text, 'en'

    # Prioritize Bangla/Banglish detection
    if is_bangla(text):
        target_lang = 'bn'
    else:
        try:
            target_lang = detect(text)
        except Exception:
            target_lang = 'en'

    if target_lang == 'en':
        return text, 'en'

    try:
        # Use auto-detection for the actual translation to be safe
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        logger.info(f"Translated input from {target_lang} to EN: {translated}")
        return translated, target_lang
    except Exception as e:
        logger.error(f"Translation to EN failed: {e}")
        return text, 'en'

def translate_from_en(text: str, target_lang: str):
    """Translates English text back to the target language."""
    if not text or not text.strip() or target_lang == 'en':
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
    # 1. Detect language and translate to English if necessary
    message_en, original_lang = translate_to_en(request.message)

    # 2. Send the English message to Rasa
    responses = send_to_rasa(request.session_id, message_en)

    # 3. If the original message was not English, translate Rasa's responses back
    if original_lang != 'en' and isinstance(responses, list):
        for resp in responses:
            if 'text' in resp:
                resp['text'] = translate_from_en(resp['text'], original_lang)

    return responses
