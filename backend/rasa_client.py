import requests

RASA_URL = "http://localhost:5005/webhooks/rest/webhook"

def send_to_rasa(sender_id: str, message: str):
    payload = {
        "sender": sender_id,
        "message": message
    }
    try:
        res = requests.post(RASA_URL, json=payload, timeout=10)
        return res.json()
    except Exception as e:
        return {"error": str(e)}
