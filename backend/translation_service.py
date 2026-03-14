import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

# Load model and tokenizer
model_id = "facebook/nllb-200-distilled-600M"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float16 if device=="cuda" else torch.float32).to(device)

def is_bangla(text: str) -> bool:
    """Detect if text contains Bangla script or common Banglish keywords"""
    if not text:
        return False
    # Check for Bangla Unicode range
    for char in text:
        if '\u0980' <= char <= '\u09FF':
            return True
    # Expanded Banglish words
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

def translate_to_english(text, source_lang="ben_Beng"):
    tokenizer.src_lang = source_lang
    inputs = tokenizer(text, return_tensors="pt").to(device)
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["eng_Latn"], max_length=256
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def translate_to_bangla(text):
    tokenizer.src_lang = "eng_Latn"
    inputs = tokenizer(text, return_tensors="pt").to(device)
    translated_tokens = model.generate(
        **inputs, forced_bos_token_id=tokenizer.lang_code_to_id["ben_Beng"], max_length=256
    )
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
