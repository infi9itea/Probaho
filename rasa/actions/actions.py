from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
import requests
RAG_API_URL = "https://yieldingly-schizophytic-deanna.ngrok-free.dev/rag/query"
class ActionCallRAG(Action):

    def name(self) -> Text:
        return "action_call_rag"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        user_query = tracker.latest_message.get("text")

        payload = {
            "query": user_query
        }

        try:
            response = requests.post(RAG_API_URL, json=payload, timeout=20)
            response.raise_for_status()
            result = response.json()

            answer = result.get(
                "answer",
                "I don't have that information right now."
            )

        except Exception as e:
            answer = "I'm having trouble accessing university information right now."

        dispatcher.utter_message(text=answer)
        return []
