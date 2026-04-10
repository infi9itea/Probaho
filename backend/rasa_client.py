import requests
import os

RASA_URL = os.getenv("RASA_URL", "http://localhost:5005/webhooks/rest/webhook")

def send_to_rasa(sender_id: str, message: str):
    payload = {
        "sender": sender_id,
        "message": message
    }
    try:
        res = requests.post(RASA_URL, json=payload, timeout=60)
        return res.json()
    except Exception as e:
        return [{"text": f"Error: {str(e)}"}]
