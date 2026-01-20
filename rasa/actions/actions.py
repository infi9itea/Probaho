import os
import requests
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from typing import Any, Text, Dict, List

RAG_API_URL = os.getenv("RAG_API_URL")
if not RAG_API_URL:
    raise RuntimeError("RAG_API_URL environment variable not set")

class ActionRagQuery(Action):

    def name(self) -> Text:
        return "action_rag_query"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        # 1. Get user message
        user_query = tracker.latest_message.get("text")

        # 2. Call RAG API
        try:
            response = requests.post(
                RAG_API_URL,
                json={"query": user_query},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            answer = data.get("answer", "Sorry, I could not find an answer.")

        except Exception:
            answer = "Sorry, the information service is currently unavailable."

        # 3. Send answer back to user
        dispatcher.utter_message(text=answer)

        return []
