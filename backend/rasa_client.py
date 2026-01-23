import requests

RASA_URL = "http://rasa:5005/webhooks/rest/webhook"  # use 'rasa' service name from docker-compose

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
