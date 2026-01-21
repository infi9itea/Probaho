import requests
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
RAG_API_URL = "https://YOUR_NGROK_URL/rag/query"
class ActionCallRAG(Action):
    def name(self):
        return "action_call_rag"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain):

        user_message = tracker.latest_message.get("text")

        if not user_message:
            dispatcher.utter_message(text="Please ask a question.")
            return []

        payload = {
            "query": user_message
        }

        try:
            response = requests.post(
                RAG_API_URL,
                json=payload,
                timeout=15
            )

            if response.status_code != 200:
                dispatcher.utter_message(
                    text="Sorry, I'm having trouble retrieving information."
                )
                return []

            data = response.json()
            answer = data.get("answer", "")
            confidence = data.get("confidence", 0)

            if confidence < 0.2 or not answer:
                dispatcher.utter_message(response="utter_no_info")
            else:
                dispatcher.utter_message(text=answer)

        except Exception:
            dispatcher.utter_message(
                text="The information service is currently unavailable."
            )

        return []
