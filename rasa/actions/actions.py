import os
import json
import requests
import logging
import aiohttp
import asyncio
from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import UserUtteranceReverted, SlotSet

logger = logging.getLogger(__name__)

class ActionPhi3RagAnswer(Action):
    """RAG-powered answer generation using TinyLlama"""
    def name(self) -> Text:
        # change the action name here
        return "action_call_rag"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        user_message = tracker.latest_message.get("text", "")

        if not user_message.strip():
            dispatcher.utter_message(text="I didn't catch that. Could you please rephrase?")
            return []

        # Build conversation history (for future use)
        history = self._build_history(tracker)

        # Call RAG service with TinyLlama
        answer, confidence, sources, processing_time = self._call_rag(user_message)

        # Send appropriate response based on confidence
        self._send_response(dispatcher, answer, confidence, sources, processing_time)

        return []

    def _call_rag(self, query: str) -> tuple:
        """
        Call RAG service (TinyLlama-powered) and return structured response

        Returns:
            (answer, confidence, sources, processing_time)
        """
        try:
            # FIXED: Only send what the API accepts
            payload = {
                "query": query,
                "top_k": 20  # Increased for better context
            }

            #logger.info(f"RAG Query: {query[:50]}...")
            RAG_API_URL = os.environ.get("RAG_API_URL", "https://yieldingly-schizophytic-deanna.ngrok-free.dev/rag/query")
            headers = {"Ngrok-Skip-Browser-Warning": "true"}
            response = requests.post(RAG_API_URL, json=payload, headers=headers, timeout=500)

            if response.status_code == 200:
                data = response.json()
                logger.info(f"RAG Response Raw: {data}")

                # FIXED: Use correct field names from FastAPI QueryResponse
                answer = data.get("response", "")  # NOT "answer"
                confidence = float(data.get("confidence", 0.0))
                sources = data.get("sources", [])
                processing_time = data.get("processing_time", 0.0)

                logger.info(f"Parsed - Answer: '{answer[:100]}...', Confidence: {confidence}")
                logger.info(
                    f"RAG response received - Confidence: {confidence:.2f}, "
                    f"Time: {processing_time:.2f}s, Sources: {len(sources)}"
                )

                return answer, confidence, sources, processing_time

            else:
                logger.error(f"RAG service error: {response.status_code} - {response.text}")
                return (
                    "I'm having trouble connecting to my knowledge base.",
                    0.0,
                    [],
                    0.0
                )

        except requests.Timeout:
            logger.error("RAG service timeout")
            return (
                "The request is taking too long. Please try again.",
                0.0,
                [],
                0.0
            )
        except requests.ConnectionError:
            logger.error("RAG service connection error")
            return (
                "I can't connect to the answer service. Please try again later.",
                0.0,
                [],
                0.0
            )
        except Exception as e:
            logger.exception(f"Unexpected RAG error: {e}")
            return (
                "Something went wrong while processing your question.",
                0.0,
                [],
                0.0
            )

    # Replace the _send_response confidence branching with this clearer logic:

    def _send_response(
        self,
        dispatcher: CollectingDispatcher,
        answer: str,
        confidence: float,
        sources: List[str],
        processing_time: float
    ):
        """
        Confidence levels:
        - >= 0.7: High confidence - send answer
        - 0.4-0.7: Medium confidence - send answer with disclaimer
        - < 0.4: Low confidence - send answer with strong disclaimer
        """
        if not answer:
            dispatcher.utter_message(
                text=(
                    "I'm sorry, I couldn't find a reliable answer for that. "
                    "For accurate details, please contact:\nadmissions@ewubd.edu\n"
                )
            )
            return

        if confidence < 0.4:
            # Low confidence
            dispatcher.utter_message(
                text=f"{answer}\n\nNote: This information is generated and might not be 100% accurate. Please verify with admissions@ewubd.edu"
            )
        elif confidence < 0.7:
            # Medium confidence
            dispatcher.utter_message(
                text=f"{answer}\n\nNote: Please verify this information with admissions@ewubd.edu"
            )
        else:
            # High confidence
            dispatcher.utter_message(text=answer)

        # Add up to top-2 sources for transparency (if provided)
        if sources:
            unique_sources = sorted({s for s in sources if s})[:2]
            if unique_sources:
                source_names = [s.replace('.json', '').replace('_', ' ').title() for s in unique_sources]
                dispatcher.utter_message(text=f"Source: {', '.join(source_names)}")

        # Log performance (for monitoring)
        logger.info(f"Response sent - Confidence: {confidence:.2f}, Time: {processing_time:.2f}s")

    def _build_history(self, tracker: Tracker) -> str:
        """
        Extract last 3 conversation turns (for future use)

        Note: Current RAG API doesn't use history yet, but keeping
        this for potential future enhancement
        """
        events = tracker.events
        history = []

        for event in reversed(events[-12:]):  # Look at more events
            if event.get("event") == "user":
                text = event.get("text", "")
                if text and text.strip():
                    history.append(f"User: {text}")
            elif event.get("event") == "bot":
                text = event.get("text", "")
                # Skip system messages
                if text and not text.startswith(("", "", "")):
                    history.append(f"Bot: {text}")

            # Stop once we have 3 full exchanges
            if len(history) >= 6:
                break

        return "\n".join(reversed(history[-6:]))

def call_rag_fallback(dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
    """Helper function to call RAG fallback from other actions"""
    rag_action = ActionPhi3RagAnswer()
    return rag_action.run(dispatcher, tracker, domain)

class ActionDefaultFallback(Action):
    """Fallback to RAG for unrecognized intents"""
    def name(self) -> Text:
        return "action_default_fallback"
    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        user_message = tracker.latest_message.get("text", "")
        logger.info(f"Fallback triggered for: {user_message}")
        # Try RAG for unrecognized intents
        rag_action = ActionPhi3RagAnswer()
        return rag_action.run(dispatcher, tracker, domain)






# ========================================
# HELPER FUNCTIONS TO LOAD ALL DATA
# ========================================

BASE_URL = "https://raw.githubusercontent.com/Atkiya/RasaChatbot/main/"

def load_json_file(filename):
    try:
        url = BASE_URL + filename
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"ERROR: Failed to fetch {filename}. Status: {response.status_code}")
            return None
    except Exception as e:
        print(f"ERROR loading {filename}: {str(e)}")
        return None

def load_admission_calendar():
    return load_json_file("dynamic_admission_calendar.json")

def load_admission_process():
    return load_json_file("dynamic_admission_process.json")

def load_admission_requirements():
    return load_json_file("dynamic_admission_requirements.json")

def load_tuition_fees():
    return load_json_file("dynamic_tution_fees.json")

def load_events_workshops():
    return load_json_file("dynamic_events_workshops.json")

def load_faculty():
    return load_json_file("dynamic_faculty.json")

def load_grading():
    return load_json_file("dynamic_grading.json")

def load_facilities():
    return load_json_file("dynamic_facilites.json")

# ========================================
# STATIC FILES
# ========================================
def load_about_ewu():
    return load_json_file("static_aboutEWU.json")

def load_admin():
    return load_json_file("static_Admin.json")

def load_all_programs():
    return load_json_file("static_AllAvailablePrograms.json")

def load_campus_life():
    return load_json_file("static_campus_life.json")

def load_career_counseling():
    return load_json_file("static_Career_Counseling_Center.json")

def load_clubs():
    return load_json_file("static_clubs.json")

def load_departments():
    return load_json_file("static_depts.json")

def load_facilities_static():
    return load_json_file("static_facilities.json")

def load_facilities17():
    return load_json_file("static_facilities17.json")

def load_helpdesk():
    return load_json_file("static_helpdesk.json")

def load_payment_procedure():
    return load_json_file("static_payment_procedure.json")

def load_policy():
    return load_json_file("static_Policy.json")

def load_programs():
    return load_json_file("static_Programs.json")

def load_rules():
    return load_json_file("static_Rules.json")

def load_scholarships():
    return load_json_file("static_scholarship_and_financial.json")

def load_sexual_harassment():
    return load_json_file("static_Sexual_harassment.json")

def load_tuition_fees_static():
    return load_json_file("static_Tuition_fees.json")

# ========================================
# GRADUATE PROGRAMS
# ========================================
def load_ma_english():
    return load_json_file("ma_english.json")

def load_mba_emba():
    return load_json_file("mba_emba.json")

def load_mds():
    return load_json_file("mds.json")

def load_mphil_pharmacy():
    return load_json_file("mphil_pharmacy.json")

def load_mss_economics():
    return load_json_file("mss_eco.json")

def load_ms_cse():
    return load_json_file("ms_cse.json")

def load_ms_dsa():
    return load_json_file("ms_dsa.json")

def load_tesol():
    return load_json_file("tesol.json")

# ========================================
# UNDERGRADUATE PROGRAMS
# ========================================
def load_st_ba():
    return load_json_file("st_ba.json")

def load_st_ce():
    return load_json_file("st_ce.json")

def load_st_cse():
    return load_json_file("st_cse.json")

def load_st_ece():
    return load_json_file("st_ece.json")

def load_st_economics():
    return load_json_file("st_economics.json")

def load_st_eee():
    return load_json_file("st_eee.json")

def load_st_english():
    return load_json_file("st_english.json")

def load_st_geb():
    return load_json_file("st_geb.json")

def load_st_information_studies():
    return load_json_file("st_information_studies.json")

def load_st_law():
    return load_json_file("st_law.json")

def load_st_math():
    return load_json_file("st_math.json")

def load_st_pharmacy():
    return load_json_file("st_pharmacy.json")

def load_st_social_relations():
    return load_json_file("st_social_relations.json")

def load_st_sociology():
    return load_json_file("st_sociology.json")

# ========================================
# HELPER FUNCTION TO LOAD ALL FILES
# ========================================
def load_all_knowledge_base():
    """Load all JSON files at once into a dictionary"""
    return {
        # Dynamic files
        "admission_calendar": load_admission_calendar(),
        "admission_process": load_admission_process(),
        "admission_requirements": load_admission_requirements(),
        "tuition_fees": load_tuition_fees(),
        "events_workshops": load_events_workshops(),
        "faculty": load_faculty(),
        "grading": load_grading(),
        "facilities": load_facilities(),
        
        # Static files
        "about_ewu": load_about_ewu(),
        "admin": load_admin(),
        "all_programs": load_all_programs(),
        "campus_life": load_campus_life(),
        "career_counseling": load_career_counseling(),
        "clubs": load_clubs(),
        "departments": load_departments(),
        "facilities_static": load_facilities_static(),
        "facilities17": load_facilities17(),
        "helpdesk": load_helpdesk(),
        "payment_procedure": load_payment_procedure(),
        "policy": load_policy(),
        "programs": load_programs(),
        "rules": load_rules(),
        "scholarships": load_scholarships(),
        "sexual_harassment": load_sexual_harassment(),
        "tuition_fees_static": load_tuition_fees_static(),
        
        # Graduate programs
        "ma_english": load_ma_english(),
        "mba_emba": load_mba_emba(),
        "mds": load_mds(),
        "mphil_pharmacy": load_mphil_pharmacy(),
        "mss_economics": load_mss_economics(),
        "ms_cse": load_ms_cse(),
        "ms_dsa": load_ms_dsa(),
        "tesol": load_tesol(),
        
        # Undergraduate programs
        "st_ba": load_st_ba(),
        "st_ce": load_st_ce(),
        "st_cse": load_st_cse(),
        "st_ece": load_st_ece(),
        "st_economics": load_st_economics(),
        "st_eee": load_st_eee(),
        "st_english": load_st_english(),
        "st_geb": load_st_geb(),
        "st_information_studies": load_st_information_studies(),
        "st_law": load_st_law(),
        "st_math": load_st_math(),
        "st_pharmacy": load_st_pharmacy(),
        "st_social_relations": load_st_social_relations(),
        "st_sociology": load_st_sociology(),
    }


# ========================================
# TUITION FEES (UNDERGRADUATE) ACTIONS
# ========================================

class ActionGetTuitionGeneral(Action):
    def name(self) -> Text:
        return "action_tuition_general"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        message = "**East West University Tuition Fees (Per Credit)**\n\n"
        for program in data['undergraduate_programs']['tuition_fees_per_credit']:
            message += f"- {program['program']}: {program['fee_per_credit']:,} BDT/credit\n"
        
        message += f"\n*Applicable from: {data['page_info']['applicable_from']}*"
        dispatcher.utter_message(text=message)
        return []

class ActionGetApplicationFee(Action):
    def name(self) -> Text:
        return "action_application_fee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        app_fee = data['fee_categories']['application_fee']
        message = f"The application fee at East West University is **{app_fee}** (online processing fee, non-refundable)."
        dispatcher.utter_message(text=message)
        return []

class ActionGetTuitionCSE(Action):
    def name(self) -> Text:
        return "action_tuition_cse"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_programs']['detailed_fee_structure']
                    if 'CSE' in p['program']), None)
        if prog:
            message = (f"**B.Sc. in Computer Science & Engineering (CSE)**\n\n"
                      f" **Tuition Fee:** {prog['tuition_fees']:,} BDT\n"
                      f" **Total Credits:** {prog['credits']}\n"
                      f" **Total Program Cost:** {prog['grand_total']:,} BDT")
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionGetTuitionBBA(Action):
    def name(self) -> Text:
        return "action_tuition_bba"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_programs']['detailed_fee_structure']
                    if p['program'] == 'BBA'), None)
        if prog:
            message = (f"**BBA (Bachelor of Business Administration)**\n\n"
                      f" **Tuition Fee:** {prog['tuition_fees']:,} BDT\n"
                      f" **Total Credits:** {prog['credits']}\n"
                      f" **Total Program Cost:** {prog['grand_total']:,} BDT")
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionGetTuitionEconomics(Action):
    def name(self) -> Text:
        return "action_tuition_economics"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_programs']['detailed_fee_structure']
                    if 'Economics' in p['program']), None)
        if prog:
            message = (f"**BSS in Economics**\n\n"
                      f" **Tuition Fee:** {prog['tuition_fees']:,} BDT\n"
                      f" **Total Credits:** {prog['credits']}\n"
                      f" **Total Program Cost:** {prog['grand_total']:,} BDT")
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionGetTuitionEnglish(Action):
    def name(self) -> Text:
        return "action_tuition_english"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_programs']['detailed_fee_structure']
                    if 'English' in p['program']), None)
        if prog:
            message = (f"**BA in English**\n\n"
                      f" **Tuition Fee:** {prog['tuition_fees']:,} BDT\n"
                      f" **Total Credits:** {prog['credits']}\n"
                      f" **Total Program Cost:** {prog['grand_total']:,} BDT")
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionGetTuitionLaw(Action):
    def name(self) -> Text:
        return "action_tuition_law"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_programs']['detailed_fee_structure']
                    if 'LL.B' in p['program']), None)
        if prog:
            message = (f"**LL.B (Honours)**\n\n"
                      f" **Tuition Fee:** {prog['tuition_fees']:,} BDT\n"
                      f" **Total Credits:** {prog['credits']}\n"
                      f" **Total Program Cost:** {prog['grand_total']:,} BDT")
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionGetTuitionSociology(Action):
    def name(self) -> Text:
        return "action_tuition_sociology"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_programs']['detailed_fee_structure']
                    if 'Sociology' in p['program']), None)
        if prog:
            message = (f"**BSS in Sociology**\n\n"
                      f" **Tuition Fee:** {prog['tuition_fees']:,} BDT\n"
                      f" **Total Credits:** {prog['credits']}\n"
                      f" **Total Program Cost:** {prog['grand_total']:,} BDT")
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionGetTuitionInformationStudies(Action):
    def name(self) -> Text:
        return "action_tuition_information_studies"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_programs']['detailed_fee_structure']
                    if 'Information Studies' in p['program']), None)
        if prog:
            message = (f"**BSS in Information Studies**\n\n"
                      f" **Tuition Fee:** {prog['tuition_fees']:,} BDT\n"
                      f" **Total Credits:** {prog['credits']}\n"
                      f" **Total Program Cost:** {prog['grand_total']:,} BDT")
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionGetTuitionICE(Action):
    def name(self) -> Text:
        return "action_tuition_ice"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_programs']['detailed_fee_structure']
                    if 'ICE' in p['program'] or 'Communication' in p['program']), None)
        if prog:
            message = (f"**B.Sc. in Information & Communication Engineering (ICE)**\n\n"
                      f" **Tuition Fee:** {prog['tuition_fees']:,} BDT\n"
                      f" **Total Credits:** {prog['credits']}\n"
                      f" **Total Program Cost:** {prog['grand_total']:,} BDT")
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionGetTuitionEEE(Action):
    def name(self) -> Text:
        return "action_tuition_eee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_programs']['detailed_fee_structure']
                    if 'EEE' in p['program']), None)
        if prog:
            message = (f"**B.Sc. in Electrical & Electronic Engineering (EEE)**\n\n"
                      f" **Tuition Fee:** {prog['tuition_fees']:,} BDT\n"
                      f" **Total Credits:** {prog['credits']}\n"
                      f" **Total Program Cost:** {prog['grand_total']:,} BDT")
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionGetTuitionPharmacy(Action):
    def name(self) -> Text:
        return "action_tuition_pharmacy"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_programs']['detailed_fee_structure']
                    if 'Pharm' in p['program']), None)
        if prog:
            message = (f"**Bachelor of Pharmacy (B.Pharm)**\n\n"
                      f" **Tuition Fee:** {prog['tuition_fees']:,} BDT\n"
                      f" **Total Credits:** {prog['credits']}\n"
                      f" **Total Program Cost:** {prog['grand_total']:,} BDT")
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionGetTuitionGEB(Action):
    def name(self) -> Text:
        return "action_tuition_geb"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_programs']['detailed_fee_structure']
                    if 'GEB' in p['program'] or 'Genetic' in p['program']), None)
        if prog:
            message = (f"**B.Sc. in Genetic Engineering & Biotechnology (GEB)**\n\n"
                      f" **Tuition Fee:** {prog['tuition_fees']:,} BDT\n"
                      f" **Total Credits:** {prog['credits']}\n"
                      f" **Total Program Cost:** {prog['grand_total']:,} BDT")
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionGetTuitionCivil(Action):
    def name(self) -> Text:
        return "action_tuition_civil"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_programs']['detailed_fee_structure']
                    if 'Civil' in p['program']), None)
        if prog:
            message = (f"**B.Sc. in Civil Engineering**\n\n"
                      f" **Tuition Fee:** {prog['tuition_fees']:,} BDT\n"
                      f" **Total Credits:** {prog['credits']}\n"
                      f" **Total Program Cost:** {prog['grand_total']:,} BDT")
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionGetTuitionPPHS(Action):
    def name(self) -> Text:
        return "action_tuition_pphs"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_programs']['detailed_fee_structure']
                    if 'PPHS' in p['program'] or 'Public Health' in p['program']), None)
        if prog:
            message = (f"**BSS in Population & Public Health Sciences (PPHS)**\n\n"
                      f" **Tuition Fee:** {prog['tuition_fees']:,} BDT\n"
                      f" **Total Credits:** {prog['credits']}\n"
                      f" **Total Program Cost:** {prog['grand_total']:,} BDT")
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionGetTuitionMath(Action):
    def name(self) -> Text:
        return "action_tuition_math"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_programs']['detailed_fee_structure']
                    if 'Mathematics' in p['program']), None)
        if prog:
            message = (f"**B.Sc. in Mathematics**\n\n"
                      f" **Tuition Fee:** {prog['tuition_fees']:,} BDT\n"
                      f" **Total Credits:** {prog['credits']}\n"
                      f" **Total Program Cost:** {prog['grand_total']:,} BDT")
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionGetTuitionDataScience(Action):
    def name(self) -> Text:
        return "action_tuition_data_science"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_programs']['detailed_fee_structure']
                    if 'Data Science' in p['program']), None)
        if prog:
            message = (f"**B.Sc. in Data Science & Analytics**\n\n"
                      f" **Tuition Fee:** {prog['tuition_fees']:,} BDT\n"
                      f" **Total Credits:** {prog['credits']}\n"
                      f" **Total Program Cost:** {prog['grand_total']:,} BDT")
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionGetTuitionSocialRelations(Action):
    def name(self) -> Text:
        return "action_tuition_social_relations"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_programs']['detailed_fee_structure']
                    if 'Social Relations' in p['program']), None)
        if prog:
            message = (f"**BSS in Social Relations**\n\n"
                      f" **Tuition Fee:** {prog['tuition_fees']:,} BDT\n"
                      f" **Total Credits:** {prog['credits']}\n"
                      f" **Total Program Cost:** {prog['grand_total']:,} BDT")
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)
    
# ========================================
# MISSING TUITION FEES (GRADUATE) ACTIONS
# ========================================

class ActionMSDataScienceFee(Action):
    def name(self) -> Text:
        return "action_ms_data_science_fee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        prog = next((p for p in data['graduate_programs']['detailed_fee_structure']
                    if 'Data Science' in p['program'] and 'Analytics' in p['program']), None)
        if prog:
            message = (f"**M.S. in Data Science and Analytics**\n\n"
                      f" **Tuition Fee:** {prog['tuition_fees']:,} BDT\n"
                      f" **Total Credits:** {prog['credits']}\n"
                      f" **Total Program Cost:** {prog['grand_total']:,} BDT")
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)
# ========================================
# MISSING GRADUATE PROGRAM ACTIONS
# ========================================

class ActionMAEnglishExtendedFee(Action):
    def name(self) -> Text:
        return "action_ma_english_extended_fee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        prog = next((p for p in data['graduate_programs']['detailed_fee_structure']
                    if p['program'] == 'MA in English (Extended)'), None)
        if prog:
            message = (f"**MA in English (Extended - 45 Credits)**\n\n"
                      f" **Tuition Fee:** {prog['tuition_fees']:,} BDT\n"
                      f" **Total Credits:** {prog['credits']}\n"
                      f" **Total Program Cost:** {prog['grand_total']:,} BDT")
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionMATESOL42Fee(Action):
    def name(self) -> Text:
        return "action_ma_tesol_42_fee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        prog = next((p for p in data['graduate_programs']['detailed_fee_structure']
                    if p['program'] == 'MA in TESOL' and p['credits'] == 42), None)
        if prog:
            message = (f"**MA in TESOL (42 Credits)**\n\n"
                      f" **Tuition Fee:** {prog['tuition_fees']:,} BDT\n"
                      f" **Total Credits:** {prog['credits']}\n"
                      f" **Total Program Cost:** {prog['grand_total']:,} BDT")
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionMATESOL48Fee(Action):
    def name(self) -> Text:
        return "action_ma_tesol_48_fee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        prog = next((p for p in data['graduate_programs']['detailed_fee_structure']
                    if p['program'] == 'MA in TESOL' and p['credits'] == 48), None)
        if prog:
            message = (f"**MA in TESOL (48 Credits)**\n\n"
                      f" **Tuition Fee:** {prog['tuition_fees']:,} BDT\n"
                      f" **Total Credits:** {prog['credits']}\n"
                      f" **Total Program Cost:** {prog['grand_total']:,} BDT")
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionMATESOL40Fee(Action):
    def name(self) -> Text:
        return "action_ma_tesol_40_fee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        prog = next((p for p in data['graduate_programs']['detailed_fee_structure']
                    if p['program'] == 'MA in TESOL' and p['credits'] == 40), None)
        if prog:
            message = (f"**MA in TESOL (40 Credits)**\n\n"
                      f" **Tuition Fee:** {prog['tuition_fees']:,} BDT\n"
                      f" **Total Credits:** {prog['credits']}\n"
                      f" **Total Program Cost:** {prog['grand_total']:,} BDT")
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

# ========================================
# DIPLOMA PROGRAMS ACTIONS
# ========================================

class ActionPPDMDiplomaFee(Action):
    def name(self) -> Text:
        return "action_ppdm_diploma_fee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        prog = next((p for p in data['diploma_programs']['detailed_fee_structure']
                    if 'PPDM' in p['program'] or 'Disaster Management' in p['program']), None)
        if prog:
            message = (f"**{prog['program']}**\n\n"
                      f" **Tuition Fee:** {prog['tuition_fees']:,} BDT\n"
                      f" **Total Credits:** {prog['credits']}\n"
                      f" **Total Program Cost:** {prog['grand_total']:,} BDT")
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

# ========================================
# COMPREHENSIVE TUITION BREAKDOWN (ALL PROGRAMS)
# ========================================

class ActionCompleteTuitionStructure(Action):
    def name(self) -> Text:
        return "action_complete_tuition_structure"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_tuition_fees()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        message = "**Complete Tuition Fee Structure at EWU**\n\n"
        message += "**UNDERGRADUATE PROGRAMS** (15 programs)\n"
        for prog in data['undergraduate_programs']['detailed_fee_structure']:
            message += f"- {prog['program']}: {prog['grand_total']:,} BDT (Total)\n"
        
        message += "\n**GRADUATE PROGRAMS** (13 programs)\n"
        for prog in data['graduate_programs']['detailed_fee_structure']:
            message += f"- {prog['program']}: {prog['grand_total']:,} BDT (Total)\n"
        
        message += "\n**DIPLOMA PROGRAMS** (1 program)\n"
        for prog in data['diploma_programs']['detailed_fee_structure']:
            message += f"- {prog['program']}: {prog['grand_total']:,} BDT (Total)\n"
        
        dispatcher.utter_message(text=message)
        return []


# ========================================
# TUITION FEES (GRADUATE) ACTIONS
# ========================================

class ActionMBAFee(Action):
    def name(self) -> Text:
        return "action_mba_fee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="**MBA Program Fee**\n\nFor detailed MBA tuition information, please contact the admissions office at admissions@ewubd.edu or call 09666775577.")
        return []

class ActionEMBAFee(Action):
    def name(self) -> Text:
        return "action_emba_fee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="**Executive MBA (EMBA) Program Fee**\n\nFor detailed EMBA tuition information, please contact the admissions office at admissions@ewubd.edu or call 09666775577.")
        return []

class ActionMDSFee(Action):
    def name(self) -> Text:
        return "action_mds_fee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="**Master in Development Studies (MDS) Program Fee**\n\nFor detailed MDS tuition information, please contact admissions.")
        return []

class ActionMSSEconomicsFee(Action):
    def name(self) -> Text:
        return "action_mss_economics_fee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="**MSS in Economics Program Fee**\n\nFor detailed MSS Economics tuition information, please contact admissions.")
        return []

class ActionMAEnglishFee(Action):
    def name(self) -> Text:
        return "action_ma_english_fee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="**MA in English Program Fee**\n\nFor detailed MA English tuition information, please contact admissions.")
        return []

class ActionMATESOLFee(Action):
    def name(self) -> Text:
        return "action_ma_tesol_fee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="**MA in TESOL Program Fee**\n\nFor detailed MA TESOL tuition information, please contact admissions.")
        return []

class ActionLLMFee(Action):
    def name(self) -> Text:
        return "action_llm_fee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="**Master of Laws (LL.M.) Program Fee**\n\nFor detailed LL.M tuition information, please contact admissions.")
        return []

class ActionMPRHGDFee(Action):
    def name(self) -> Text:
        return "action_mprhgd_fee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="**MPRHGD Program Fee**\n\nFor detailed MPRHGD tuition information, please contact admissions.")
        return []

class ActionDSAnalyticsFee(Action):
    def name(self) -> Text:
        return "action_ds_analytics_fee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="**MS Data Science & Analytics Program Fee**\n\nFor detailed MS Data Science tuition information, please contact admissions.")
        return []

class ActionMSCSEFee(Action):
    def name(self) -> Text:
        return "action_ms_cse_fee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="**MS in Computer Science & Engineering (MS CSE) Program Fee**\n\nFor detailed MS CSE tuition information, please contact admissions.")
        return []

class ActionMPharmFee(Action):
    def name(self) -> Text:
        return "action_mpharm_fee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="**M.Pharm Program Fee**\n\nFor detailed M.Pharm tuition information, please contact admissions.")
        return []

class ActionPGDEDFee(Action):
    def name(self) -> Text:
        return "action_pgded_fee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="**PGDED (Post Graduate Diploma in Entrepreneurship Development) Program Fee**\n\nFor detailed PGDED tuition information, please contact admissions.")
        return []

class ActionPPDMFee(Action):
    def name(self) -> Text:
        return "action_ppdm_fee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="**PPDM (Post Graduate Diploma in Population, Public Health and Disaster Management) Program Fee**\n\nFor detailed PPDM tuition information, please contact admissions.")
        return []

# ========================================
# ADMISSION DEADLINES
# ========================================

class ActionAdmissionDeadlineGeneral(Action):
    def name(self) -> Text:
        return "action_admission_deadline_general"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        # Build header message
        message = f"**{data['page_info']['semester']} Admission Deadlines**\n\n"
        
        # Undergraduate Programs Section
        message += "** Undergraduate Programs:**\n"
        for program in data['undergraduate_admission']:
            prog_name = program['program']
            deadline = program['application_deadline']
            message += f"• {prog_name}: {deadline}\n"
        
        # Graduate Programs Section
        message += "\n** Graduate Programs:**\n"
        for program in data['graduate_admission']:
            prog_name = program['program']
            deadline = program['application_deadline']
            message += f"• {prog_name}: {deadline}\n"
        
        # Footer note
        message += f"\n*{data['page_info']['disclaimer']}*"
        
        dispatcher.utter_message(text=message)
        return []


class ActionAdmissionDeadlineCSE(Action):
    def name(self) -> Text:
        return "action_admission_deadline_cse"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_admission'] if 'CSE' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Application Deadline:** {prog['application_deadline']}\n📝 **Test Date:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlineBBA(Action):
    def name(self) -> Text:
        return "action_admission_deadline_bba"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_admission'] if 'BBA' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Application Deadline:** {prog['application_deadline']}\n📝 **Test Date:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlineEconomics(Action):
    def name(self) -> Text:
        return "action_admission_deadline_economics"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_admission'] if 'Economics' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlineEnglish(Action):
    def name(self) -> Text:
        return "action_admission_deadline_english"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_admission'] if 'English' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlineLaw(Action):
    def name(self) -> Text:
        return "action_admission_deadline_law"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_admission'] if 'LLB' in p['program'] or 'Law' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlineSociology(Action):
    def name(self) -> Text:
        return "action_admission_deadline_sociology"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_admission'] if 'Sociology' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlineInformationStudies(Action):
    def name(self) -> Text:
        return "action_admission_deadline_information_studies"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_admission'] if 'Information Studies' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlinePPHS(Action):
    def name(self) -> Text:
        return "action_admission_deadline_pphs"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_admission'] if 'Public Health' in p['program'] or 'PPHS' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlineICE(Action):
    def name(self) -> Text:
        return "action_admission_deadline_ice"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_admission'] if 'ICE' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlineEEE(Action):
    def name(self) -> Text:
        return "action_admission_deadline_eee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_admission'] if 'EEE' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlinePharmacy(Action):
    def name(self) -> Text:
        return "action_admission_deadline_pharmacy"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_admission'] if 'Pharmacy' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlineGEB(Action):
    def name(self) -> Text:
        return "action_admission_deadline_geb"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_admission'] if 'Genetic' in p['program'] or 'Biotechnology' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlineCivil(Action):
    def name(self) -> Text:
        return "action_admission_deadline_civil"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_admission'] if 'Civil' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlineMath(Action):
    def name(self) -> Text:
        return "action_admission_deadline_math"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_admission'] if 'Mathematics' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlineDataScience(Action):
    def name(self) -> Text:
        return "action_admission_deadline_data_science"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_admission'] if 'Data Science' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

# ========================================
# ADMISSION DEADLINES (GRADUATE)
# ========================================

class ActionAdmissionDeadlineMBA(Action):
    def name(self) -> Text:
        return "action_admission_deadline_mba"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['graduate_admission'] if 'MBA' in p['program'] and 'Executive' not in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlineEMBA(Action):
    def name(self) -> Text:
        return "action_admission_deadline_emba"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['graduate_admission'] if 'Executive MBA' in p['program'] or 'EMBA' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlineMDS(Action):
    def name(self) -> Text:
        return "action_admission_deadline_mds"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['graduate_admission'] if 'MDS' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlineMSSEconomics(Action):
    def name(self) -> Text:
        return "action_admission_deadline_mss_economics"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['graduate_admission'] if 'MSS' in p['program'] and 'Economics' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlineMAEnglish(Action):
    def name(self) -> Text:
        return "action_admission_deadline_ma_english"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['graduate_admission'] if 'MA in English' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlineMATESOL(Action):
    def name(self) -> Text:
        return "action_admission_deadline_ma_tesol"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['graduate_admission'] if 'TESOL' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlineMPRHGD(Action):
    def name(self) -> Text:
        return "action_admission_deadline_mprhgd"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['graduate_admission'] if 'MPRHGD' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlineLLM(Action):
    def name(self) -> Text:
        return "action_admission_deadline_llm"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['graduate_admission'] if 'LLM' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlineMSCSE(Action):
    def name(self) -> Text:
        return "action_admission_deadline_ms_cse"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['graduate_admission'] if 'MS in CSE' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlineMSDataScience(Action):
    def name(self) -> Text:
        return "action_admission_deadline_ms_data_science"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['graduate_admission'] if 'Data Science' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlineMPharm(Action):
    def name(self) -> Text:
        return "action_admission_deadline_mpharm"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['graduate_admission'] if 'Master of Pharmacy' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionDeadlinePPDM(Action):
    def name(self) -> Text:
        return "action_admission_deadline_ppdm"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['graduate_admission'] if 'PPDM' in p['program']), None)
        if prog:
            message = f"**{prog['program']}**\n\n📅 **Deadline:** {prog['application_deadline']}\n📝 **Test:** {prog['admission_test']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

# ========================================
# ADMISSION TEST DATES
# ========================================

class ActionAdmissionTestDateGeneral(Action):
    def name(self) -> Text:
        return "action_admission_test_date_general"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        message = "**Admission Test Dates at EWU**\n\n"
        message += "**Engineering/Science Programs:** Aug 30, 2025 at 2:30 PM\n"
        message += "**Business/Arts Programs:** Aug 30, 2025 at 10:00 AM\n\n"
        message += "*For specific program test dates, please ask about your desired program.*"
        dispatcher.utter_message(text=message)
        return []

class ActionAdmissionTestDateCSE(Action):
    def name(self) -> Text:
        return "action_admission_test_date_cse"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_admission'] if 'CSE' in p['program']), None)
        if prog:
            message = f"**Computer Science & Engineering (CSE) Admission Test**\n\n"
            message += f" **Test:** {prog['admission_test']}\n"
            message += f" **Apply by:** {prog['application_deadline']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionTestDateBBA(Action):
    def name(self) -> Text:
        return "action_admission_test_date_bba"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['undergraduate_admission'] if 'BBA' in p['program']), None)
        if prog:
            message = f"**BBA Admission Test**\n\n"
            message += f" **Test:** {prog['admission_test']}\n"
            message += f" **Apply by:** {prog['application_deadline']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionAdmissionTestDateMBA(Action):
    def name(self) -> Text:
        return "action_admission_test_date_mba"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_calendar()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prog = next((p for p in data['graduate_admission'] if 'MBA' in p['program'] and 'Executive' not in p['program']), None)
        if prog:
            message = f"**MBA Admission Test**\n\n"
            message += f" **Test:** {prog['admission_test']}\n"
            message += f" **Apply by:** {prog['application_deadline']}"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

# ========================================
# ADMISSION REQUIREMENTS ACTIONS
# ========================================

class ActionAdmissionRequirementsGeneral(Action):
    def name(self) -> Text:
        return "action_admission_requirements_general"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_requirements()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        ug = data['admission_requirements']['undergraduate']['general_programs_except_bpharm']
        message = "**Undergraduate Admission Requirements at EWU**\n\n"
        message += f" **SSC & HSC:** {ug['ssc_hsc']}\n"
        message += f" **Diploma:** {ug['diploma']}\n"
        message += f" **O/A Levels:** {ug['o_a_levels']['requirement']}\n\n"
        message += "**Admission Test Weightage:**\n"
        message += f"- Admission Test: {ug['admission_test']['weightage']['admission_test']}\n"
        message += f"- SSC/O Level: {ug['admission_test']['weightage']['ssc_o_level']}\n"
        message += f"- HSC/A Level: {ug['admission_test']['weightage']['hsc_a_level']}"
        dispatcher.utter_message(text=message)
        return []

class ActionAdmissionRequirementsCSE(Action):
    def name(self) -> Text:
        return "action_admission_requirements_cse"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_requirements()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        ug = data['admission_requirements']['undergraduate']['general_programs_except_bpharm']
        cse_req = ug['subject_requirements']['cse']
        message = "**B.Sc. in CSE Admission Requirements**\n\n"
        message += f" **Academic:** {ug['ssc_hsc']}\n"
        message += f" **Subject Requirements:** {cse_req}\n\n"
        message += "**Test Weightage:**\n"
        message += f"- Admission Test: {ug['admission_test']['weightage']['admission_test']}\n"
        message += f"- SSC: {ug['admission_test']['weightage']['ssc_o_level']}\n"
        message += f"- HSC: {ug['admission_test']['weightage']['hsc_a_level']}"
        dispatcher.utter_message(text=message)
        return []

class ActionAdmissionRequirementsPharmacy(Action):
    def name(self) -> Text:
        return "action_admission_requirements_pharmacy"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_requirements()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        pharm = data['admission_requirements']['undergraduate']['bpharm']
        message = "**B.Pharm Admission Requirements**\n\n"
        message += f"🇧🇩 **Citizenship:** {pharm['citizenship']}\n"
        message += f" **GPA:** {pharm['ssc_hsc']['aggregate']}\n"
        message += f" **Each Exam:** {pharm['ssc_hsc']['minimum_each']}\n\n"
        message += "**Subject Requirements (Minimum GPA):**\n"
        message += f"- Chemistry: {pharm['subject_gpa']['chemistry']}\n"
        message += f"- Biology: {pharm['subject_gpa']['biology']}\n"
        message += f"- Physics: {pharm['subject_gpa']['physics']}\n"
        message += f"- Mathematics: {pharm['subject_gpa']['mathematics']}\n\n"
        message += f" {pharm['special_note']}\n"
        message += f" {pharm['year_of_pass']}"
        dispatcher.utter_message(text=message)
        return []

class ActionAdmissionRequirementsMBA(Action):
    def name(self) -> Text:
        return "action_admission_requirements_mba"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_requirements()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        mba = data['admission_requirements']['graduate']['mba_emba']
        message = "**MBA Admission Requirements**\n\n"
        message += f" **Degree:** {mba['degree']}\n"
        message += f" *SSC & HSC:** {mba['ssc_hsc_gpa']}\n"
        message += f"💼 **Work Experience:** {mba['mba']['work_experience']}\n\n"
        message += "**Test Exemptions:**\n"
        message += f"- EWU Graduates: {mba['test_exemptions']['ewu_graduates']}\n"
        message += f"- Other Universities: {mba['test_exemptions']['other_universities']}"
        dispatcher.utter_message(text=message)
        return []

# ============================================================================
# MISSING ADMISSION PROCESS ACTION FUNCTIONS
# ============================================================================

class ActionAdmissionApplicationSteps(Action):
    """Display all 11 application steps for admission"""
    def name(self) -> Text:
        return "action_admission_application_steps"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_process()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        app_section = data.get('admission_process', {}).get('application', {})
        message = "** EWU Admission Application - 11 Steps**\n\n"
        
        # Website links
        links = app_section.get('website_links', [])
        message += "** Admission Websites:**\n"
        for link in links:
            message += f"- {link}\n"
        
        # Browser recommendations
        browsers = app_section.get('browser_recommendation', [])
        message += f"\n** Recommended Browsers:** {', '.join(browsers)}\n\n"
        
        # Application steps
        steps = app_section.get('steps', [])
        message += "** Application Steps:**\n\n"
    
        for step_info in steps:
            step_num = step_info.get('step', '')
            action = step_info.get('action', '')
            details = step_info.get('details', '')
            
            message += f"**Step {step_num}: {action}**\n"
            
            if isinstance(details, dict):
                for key, value in details.items():
                    message += f"- **{key}:** "
                    if isinstance(value, list):
                        message += "\n"
                        for item in value:
                            message += f"  • {item}\n"
                    else:
                        message += f"{value}\n"
            elif isinstance(details, list):
                for detail in details:
                    message += f"- {detail}\n"
            else:
                message += f"{details}\n"
            message += "\n"
        
        dispatcher.utter_message(text=message)
        return []


class ActionAdmissionContact(Action):
    """Display admission office contact information"""
    def name(self) -> Text:
        return "action_admission_contact"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_process()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        admission = data.get('admission_process', {})
        contacts = admission.get('contacts', {})
        
        message = "** EWU Admission Contact Information**\n\n"
                # Admission Office
        admission_office = contacts.get('admission_office', {})
        message += "** Admission Office**\n"
        message+= f" Address: {admission_office.get('address', 'N/A')}\n\n"
        message += "** Phone Numbers:**\n"
        for phone in admission_office.get('phone', []):
            message += f"- {phone}\n"
        message += f"Email: {admission_office.get('email', 'N/A')}\n\n"
        
        # Support Contacts
        support = contacts.get('support', {})
        message += "** Support Contacts:**\n"
        message += f" Payment Issues: {support.get('payment_issues', 'N/A')}\n"
        message += f" Technical Issues: {support.get('technical_issues', 'N/A')}\n"
        message += f" Advising/Courses: {support.get('advising_or_course_issues', 'N/A')}\n\n"
        
        # Registrar
        registrar = admission.get('registrar', {})
        message += "** Registrar**\n"
        message += f"- {registrar.get('name', 'N/A')}\n"
        message += f"- {registrar.get('designation', 'N/A')}\n"
        message += f"- {registrar.get('university', 'N/A')}"
        
        dispatcher.utter_message(text=message)
        return []


class ActionPostAdmissionGSuite(Action):
    """Display G Suite email activation instructions"""
    def name(self) -> Text:
        return "action_post_admission_g_suite"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_process()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        g_suite = data.get('admission_process', {}).get('post_admission', {}).get('g_suite_activation', {})
        
        message = "** EWU G Suite Email Activation**\n\n"
        message += f"** Important Note:** {g_suite.get('note', 'N/A')}\n\n"
        message += f"** Portal Link:** {g_suite.get('link', 'N/A')}\n\n"
        message += "**Step-by-Step Instructions:**\n"
        
        for i, instruction in enumerate(g_suite.get('instructions', []), 1):
            message += f"{i}. {instruction}\n"
        
        dispatcher.utter_message(text=message)
        return []

class ActionPostAdmissionDocumentUpload(Action):
    """Display document upload requirements"""
    def name(self) -> Text:
        return "action_post_admission_document_upload"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        data = load_admission_requirements()
        
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        required_docs = data.get('admission_requirements', {}).get('required_documents', [])
        
        if not required_docs:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        message = "** Required Documents for Admission**\n\n"
        message += f"**University:** {data.get('university', 'East West University')}\n\n"
        
        message += "** Documents You Need to Bring:**\n\n"
        for i, doc in enumerate(required_docs, 1):
            message += f"{i}. {doc}\n"
        
        message += "\n** Important Note:**\n"
        message += "• Bring both original documents and photocopies\n"
        message += "• Original documents will be returned after verification\n"
        message += "• Make sure all documents are complete and properly attested\n"
        
        dispatcher.utter_message(text=message)
        return []




class ActionPostAdmissionAdvisingSlip(Action):
    """Display advising slip access information"""
    def name(self) -> Text:
        return "action_post_admission_advising_slip"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_process()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        advising = data.get('admission_process', {}).get('post_admission', {}).get('advising_slip', {})
        
        message = "** Advising Slip Access**\n\n"
        message += f"**Purpose:** {advising.get('purpose', 'N/A')}\n\n"
        
        message += "**How to Access (4 Steps):**\n"
        for i, instruction in enumerate(advising.get('instructions', []), 1):
            message += f"{i}. {instruction}\n"
        
        message += f"\n**Academic Calendar:** {advising.get('academic_calendar_link', 'N/A')}"
        
        dispatcher.utter_message(text=message)
        return []


class ActionPostAdmissionTuitionPayment(Action):
    """Display tuition payment information"""
    def name(self) -> Text:
        return "action_post_admission_tuition_payment"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_process()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        tuition = data.get('admission_process', {}).get('post_admission', {}).get('tuition_payment', {})
        
        message = "** Tuition Payment Information**\n\n"
        
        message += "**Requirements:**\n"
        for req in tuition.get('requirements', []):
            message += f"✓ {req}\n"
        
        message += "**\n**Payment Methods:**\n"
        for method in tuition.get('payment_methods', []):
            message += f"- {method}\n"
        
        message += f"\n** Important Note:**\n{tuition.get('important_note', 'N/A')}"
        
        dispatcher.utter_message(text=message)
        return []


class ActionAdmissionImportantNotes(Action):
    """Display important admission notes"""
    def name(self) -> Text:
        return "action_admission_important_notes"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_process()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        notes = data.get('admission_process', {}).get('important_notes', [])
        
        message = "** Important Admission Notes**\n\n"
        for i, note in enumerate(notes, 1):
            message += f"{i}. {note}\n\n"
        
        dispatcher.utter_message(text=message)
        return []


class ActionCompleteAdmissionProcess(Action):
    """Display complete admission process overview"""
    def name(self) -> Text:
        return "action_complete_admission_process"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_admission_process()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        message = "** Complete EWU Admission Process**\n\n"
        
        message += "**PHASE 1: APPLICATION (11 Steps)**\n"
        message += "- Visit EWU admission website\n"
        message += "- Create account and select program\n"
        message += "- Fill application form\n"
        message += "- Pay application fee (Tk 1500)\n"
        message += "- Upload photo & signature\n"
        message += "- Submit form\n"
        message += "- Download admit card\n"
        message += "- Bring documents to exam\n\n"
        
        message += "**PHASE 2: POST-ADMISSION SETUP**\n"
        message += " Activate G Suite email account\n"
        message += " Upload required academic documents (8 docs)\n"
        message += " View advising slip with courses\n"
        message += " Pay tuition via designated banks\n\n"
        
        message += "**Need More Information?**\n"
        message += "Ask about: application steps, documents, email setup, payment methods, or contact info."
        
        dispatcher.utter_message(text=message)
        return []


class ActionAdmissionHelp(Action):
    """Help with admission queries"""
    def name(self) -> Text:
        return "action_admission_help"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        message = "** Admission Help**\n\n"
        message += "I can help you with:\n"
        message += "-  Application steps (11 steps detailed)\n"
        message += "-  Contact information\n"
        message += "-  Email setup after admission\n"
        message += "-  Document requirements\n"
        message += "-  Tuition payment methods\n"
        message += "-  Advising slip information\n"
        message += "-  Important policies\n\n"
        message += "What would you like to know?"
        
        dispatcher.utter_message(text=message)
        return []


# ========================================
# FACILITIES ACTIONS
# ========================================

class ActionFacilitiesGeneral(Action):
    def name(self) -> Text:
        return "action_facilities_general"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_facilities()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        campus = data['facilities']['campus_life']['available']
        message = "**East West University Campus Facilities**\n\n"
        for facility in campus[:7]:
            message += f" **{facility['name']}**\n{facility['description']}\n\n"
        message += "*Ask about specific facilities like library, labs, cafeteria, wifi, etc.*"
        dispatcher.utter_message(text=message)
        return []


class ActionLabFacilities(Action):
    def name(self) -> Text:
        return "action_lab_facilities"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_facilities()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        eng_labs = data['facilities']['engineering_labs']
        message = "**Engineering Laboratories at EWU**\n\n"
        message += f"**Departments:** {', '.join(eng_labs['departments'])}\n\n"
        message += "**Available Labs:**\n"
        for lab in eng_labs['labs'][:5]:
            message += f" {lab['name']}\n"
        message += f"\n*Total: {len(eng_labs['labs'])} specialized labs available*"
        dispatcher.utter_message(text=message)
        return []
    
# ========================================
# MISSING FACILITY ACTIONS
# ========================================

class ActionCampusLife(Action):
    def name(self) -> Text:
        return "action_campus_life"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_facilities()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        campus_life = data.get('facilities', {}).get('campus_life', {})
        message = f"**Campus Life**\n\n{campus_life.get('description', 'N/A')}"
        dispatcher.utter_message(text=message)
        return []

class ActionCivilEngineeringLabs(Action):
    def name(self) -> Text:
        return "action_civil_engineering_labs"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_facilities()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        labs = data.get('facilities', {}).get('civil_engineering_labs', {})
        message = f"**Civil Engineering Labs**\n\n{labs.get('description', 'N/A')}"
        dispatcher.utter_message(text=message)
        return []

class ActionEngineeringLabs(Action):
    def name(self) -> Text:
        return "action_engineering_labs"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_facilities()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        labs = data.get('facilities', {}).get('engineering_labs', {})
        message = f"**Engineering Labs**\n\n{labs.get('description', 'N/A')}"
        dispatcher.utter_message(text=message)
        return []

class ActionICSServices(Action):
    def name(self) -> Text:
        return "action_ics_services"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_facilities()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        ics = data.get('facilities', {}).get('ics_services', {})
        message = f"**ICS Services**\n\n{ics.get('description', 'N/A')}"
        dispatcher.utter_message(text=message)
        return []

class ActionPharmacyLabs(Action):
    def name(self) -> Text:
        return "action_pharmacy_labs"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_facilities()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        labs = data.get('facilities', {}).get('pharmacy_labs', {})
        message = f"**Pharmacy Labs**\n\n{labs.get('description', 'N/A')}"
        dispatcher.utter_message(text=message)
        return []

class ActionResearchCenter(Action):
    def name(self) -> Text:
        return "action_research_center"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_facilities()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        center = data.get('facilities', {}).get('research_center', {})
        message = f"**Research Center**\n\n{center.get('description', 'N/A')}"
        dispatcher.utter_message(text=message)
        return []

class ActionComputerLab(Action):
    def name(self) -> Text:
        return "action_computer_lab"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_facilities()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        lab = data.get('facilities', {}).get('computer_lab', {})
        message = f"**Computer Lab**\n\n{lab.get('description', 'N/A')}"
        dispatcher.utter_message(text=message)
        return []

class ActionWiFiInternet(Action):
    def name(self) -> Text:
        return "action_wifi_internet"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_facilities()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        wifi = data.get('facilities', {}).get('wifi_internet', {})
        message = f"**WiFi & Internet**\n\n{wifi.get('description', 'N/A')}"
        dispatcher.utter_message(text=message)
        return []

class ActionParkingFacilities(Action):
    def name(self) -> Text:
        return "action_parking_facilities"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_facilities()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        parking = data.get('facilities', {}).get('parking_facilities', {})
        message = f"**Parking Facilities**\n\n{parking.get('description', 'N/A')}"
        dispatcher.utter_message(text=message)
        return []

class ActionSportsFacilities(Action):
    def name(self) -> Text:
        return "action_sports_facilities"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_facilities()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        sports = data.get('facilities', {}).get('sports_facilities', {})
        message = f"**Sports Facilities**\n\n{sports.get('description', 'N/A')}"
        dispatcher.utter_message(text=message)
        return []

class ActionHostelFacilities(Action):
    def name(self) -> Text:
        return "action_hostel_facilities"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_facilities()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        hostel = data.get('facilities', {}).get('hostel_facilities', {})
        message = f"**Hostel Facilities**\n\n{hostel.get('description', 'N/A')}"
        dispatcher.utter_message(text=message)
        return []

class ActionMedicalFacilities(Action):
    def name(self) -> Text:
        return "action_medical_facilities"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_facilities()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        medical = data.get('facilities', {}).get('medical_facilities', {})
        message = f"**Medical Facilities**\n\n{medical.get('description', 'Yes, medical facilities are available https://www.ewubd.edu/admin-office/ewu-medical-centre')}"
        dispatcher.utter_message(text=message)
        return []

class ActionPrayerRoom(Action):
    def name(self) -> Text:
        return "action_prayer_room"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_facilities()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        prayer = data.get('facilities', {}).get('prayer_room', {})
        message = f"**Prayer Room**\n\n{prayer.get('description', 'N/A')}"
        dispatcher.utter_message(text=message)
        return []

class ActionCommonRoom(Action):
    def name(self) -> Text:
        return "action_common_room"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_facilities()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        common = data.get('facilities', {}).get('common_room', {})
        message = f"**Common Room**\n\n{common.get('description', 'N/A')}"
        dispatcher.utter_message(text=message)
        return []

class ActionCareerCounseling(Action):
    def name(self) -> Text:
        return "action_career_counseling"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_facilities()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        career = data.get('facilities', {}).get('career_counseling', {})
        message = f"**Career Counseling**\n\n{career.get('description', 'N/A')}"
        dispatcher.utter_message(text=message)
        return []
# ========================================
# MISSING FACILITY ACTIONS (4 remaining)
# ========================================

class ActionCafeteriaFacilities(Action):
    def name(self) -> Text:
        return "action_cafeteria_facilities"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_facilities()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        cafeteria = data.get('facilities', {}).get('cafeteria', {})
        message = f"**Cafeteria Facilities**\n\n{cafeteria.get('description', 'N/A')}"
        dispatcher.utter_message(text=message)
        return []

class ActionLibraryFacilities(Action):
    def name(self) -> Text:
        return "action_library_facilities"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_facilities()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        library = data.get('facilities', {}).get('library', {})
        message = f"**Library Facilities**\n\n{library.get('description', 'N/A')}"
        dispatcher.utter_message(text=message)
        return []

class ActionTransportationFacilities(Action):
    def name(self) -> Text:
        return "action_transportation_facilities"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_facilities()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        transport = data.get('facilities', {}).get('transportation', {})
        message = f"**Transportation Facilities**\n\n{transport.get('description', 'N/A')}"
        dispatcher.utter_message(text=message)
        return []


# ========================================
# EVENTS ACTIONS
# ========================================


# ========================================
# FACULTY ACTIONS
# ========================================

class ActionFacultyInfo(Action):
    def name(self) -> Text:
        return "action_faculty_info"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_faculty()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        message = "**Faculty Information - East West University**\n\n"
        message += f"**Total Departments:** {len(data.get('departments', []))}\n\n"
        message += "**Available Departments:**\n"
        for dept in list(data.get('departments', []))[:8]:
            name = dept.get('department_name', 'Unknown')
            message += f" {name}\n"
        message += "\n*As about specific department faculty (e.g., CSE faculty, BBA faculty)*"
        dispatcher.utter_message(text=message)
        return []

class ActionFacultyCSE(Action):
    def name(self) -> Text:
        return "action_faculty_cse"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_faculty()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        cse_dept = next((d for d in data.get('departments', [])
                        if 'CSE' in d.get('department_name', '') or 'Computer Science' in d.get('department_name', '')), None)
        if cse_dept and 'faculty_members' in cse_dept:
            message = f"**{cse_dept['department_name']} Faculty**\n\n"
            for faculty in cse_dept['faculty_members'][:5]:
                message += f" **{faculty.get('name', 'N/A')}**\n"
                message += f"   {faculty.get('designation', 'N/A')}\n"
                if 'email' in faculty:
                    message += f"  {faculty['email']}\n"
                message += "\n"
            total = len(cse_dept['faculty_members'])
            message += f"*Total CSE Faculty: {total} members*"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionFacultyBBA(Action):
    def name(self) -> Text:
        return "action_faculty_bba"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_faculty()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        bba_dept = next((d for d in data.get('departments', [])
                        if 'BBA' in d.get('department_name', '') or 'Business' in d.get('department_name', '')), None)
        if bba_dept and 'faculty_members' in bba_dept:
            message = f"**{bba_dept['department_name']} Faculty**\n\n"
            for faculty in bba_dept['faculty_members'][:5]:
                message += f" **{faculty.get('name', 'N/A')}**\n"
                message += f"   {faculty.get('designation', 'N/A')}\n"
                if 'email' in faculty:
                    message += f"  {faculty['email']}\n"
                message += "\n"
            total = len(bba_dept['faculty_members'])
            message += f"*Total BBA Faculty: {total} members*"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)
    
class ActionFacultyEEE(Action):
    def name(self) -> Text:
        return "action_faculty_eee"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_faculty()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        eee_dept = next((d for d in data.get('departments', [])
                        if 'EEE' in d.get('department_name', '') or 'Electrical' in d.get('department_name', '')), None)
        if eee_dept and 'faculty_members' in eee_dept:
            message = f"**{eee_dept['department_name']} Faculty**\n\n"
            for faculty in eee_dept['faculty_members'][:5]:
                message += f" **{faculty.get('name', 'N/A')}**\n"
                message += f"   {faculty.get('designation', 'N/A')}\n"
                if 'email' in faculty:
                    message += f"  {faculty['email']}\n"
                message += "\n"
            total = len(eee_dept['faculty_members'])
            message += f"*Total EEE Faculty: {total} members*"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionFacultyICE(Action):
    def name(self) -> Text:
        return "action_faculty_ice"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_faculty()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        ice_dept = next((d for d in data.get('departments', [])
                        if 'ICE' in d.get('department_name', '') or 'Communication' in d.get('department_name', '')), None)
        if ice_dept and 'faculty_members' in ice_dept:
            message = f"**{ice_dept['department_name']} Faculty**\n\n"
            for faculty in ice_dept['faculty_members'][:5]:
                message += f" **{faculty.get('name', 'N/A')}**\n"
                message += f"   {faculty.get('designation', 'N/A')}\n"
                if 'email' in faculty:
                    message += f"  {faculty['email']}\n"
                message += "\n"
            total = len(ice_dept['faculty_members'])
            message += f"*Total ICE Faculty: {total} members*"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionFacultyPharmacy(Action):
    def name(self) -> Text:
        return "action_faculty_pharmacy"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_faculty()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        pharm_dept = next((d for d in data.get('departments', [])
                          if 'Pharmacy' in d.get('department_name', '')), None)
        if pharm_dept and 'faculty_members' in pharm_dept:
            message = f"**{pharm_dept['department_name']} Faculty**\n\n"
            for faculty in pharm_dept['faculty_members'][:5]:
                message += f" **{faculty.get('name', 'N/A')}**\n"
                message += f"   {faculty.get('designation', 'N/A')}\n"
                if 'email' in faculty:
                    message += f"  {faculty['email']}\n"
                message += "\n"
            total = len(pharm_dept['faculty_members'])
            message += f"*Total Pharmacy Faculty: {total} members*"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionFacultyCivil(Action):
    def name(self) -> Text:
        return "action_faculty_civil"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_faculty()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        civil_dept = next((d for d in data.get('departments', [])
                          if 'Civil' in d.get('department_name', '')), None)
        if civil_dept and 'faculty_members' in civil_dept:
            message = f"**{civil_dept['department_name']} Faculty**\n\n"
            for faculty in civil_dept['faculty_members'][:5]:
                message += f" **{faculty.get('name', 'N/A')}**\n"
                message += f"   {faculty.get('designation', 'N/A')}\n"
                if 'email' in faculty:
                    message += f"  {faculty['email']}\n"
                message += "\n"
            total = len(civil_dept['faculty_members'])
            message += f"*Total Civil Engineering Faculty: {total} members*"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionFacultyGEB(Action):
    def name(self) -> Text:
        return "action_faculty_geb"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_faculty()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        geb_dept = next((d for d in data.get('departments', [])
                        if 'GEB' in d.get('department_name', '') or 'Genetic Engineering' in d.get('department_name', '')), None)
        if geb_dept and 'faculty_members' in geb_dept:
            message = f"**{geb_dept['department_name']} Faculty**\n\n"
            for faculty in geb_dept['faculty_members'][:5]:
                message += f" **{faculty.get('name', 'N/A')}**\n"
                message += f"   {faculty.get('designation', 'N/A')}\n"
                if 'email' in faculty:
                    message += f"  {faculty['email']}\n"
                message += "\n"
            total = len(geb_dept['faculty_members'])
            message += f"*Total GEB Faculty: {total} members*"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionFacultyEconomics(Action):
    def name(self) -> Text:
        return "action_faculty_economics"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_faculty()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        econ_dept = next((d for d in data.get('departments', [])
                         if 'Economics' in d.get('department_name', '')), None)
        if econ_dept and 'faculty_members' in econ_dept:
            message = f"**{econ_dept['department_name']} Faculty**\n\n"
            for faculty in econ_dept['faculty_members'][:5]:
                message += f" **{faculty.get('name', 'N/A')}**\n"
                message += f"   {faculty.get('designation', 'N/A')}\n"
                if 'email' in faculty:
                    message += f"  {faculty['email']}\n"
                message += "\n"
            total = len(econ_dept['faculty_members'])
            message += f"*Total Economics Faculty: {total} members*"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionFacultyEnglish(Action):
    def name(self) -> Text:
        return "action_faculty_english"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_faculty()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        eng_dept = next((d for d in data.get('departments', [])
                        if 'English' in d.get('department_name', '')), None)
        if eng_dept and 'faculty_members' in eng_dept:
            message = f"**{eng_dept['department_name']} Faculty**\n\n"
            for faculty in eng_dept['faculty_members'][:5]:
                message += f" **{faculty.get('name', 'N/A')}**\n"
                message += f"   {faculty.get('designation', 'N/A')}\n"
                if 'email' in faculty:
                    message += f"  {faculty['email']}\n"
                message += "\n"
            total = len(eng_dept['faculty_members'])
            message += f"*Total English Faculty: {total} members*"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionFacultyLaw(Action):
    def name(self) -> Text:
        return "action_faculty_law"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_faculty()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        law_dept = next((d for d in data.get('departments', [])
                        if 'Law' in d.get('department_name', '')), None)
        if law_dept and 'faculty_members' in law_dept:
            message = f"**{law_dept['department_name']} Faculty**\n\n"
            for faculty in law_dept['faculty_members'][:5]:
                message += f" **{faculty.get('name', 'N/A')}**\n"
                message += f"   {faculty.get('designation', 'N/A')}\n"
                if 'email' in faculty:
                    message += f"  {faculty['email']}\n"
                message += "\n"
            total = len(law_dept['faculty_members'])
            message += f"*Total Law Faculty: {total} members*"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionFacultyMath(Action):
    def name(self) -> Text:
        return "action_faculty_math"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_faculty()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        math_dept = next((d for d in data.get('departments', [])
                         if 'Mathematics' in d.get('department_name', '')), None)
        if math_dept and 'faculty_members' in math_dept:
            message = f"**{math_dept['department_name']} Faculty**\n\n"
            for faculty in math_dept['faculty_members'][:5]:
                message += f" **{faculty.get('name', 'N/A')}**\n"
                message += f"   {faculty.get('designation', 'N/A')}\n"
                if 'email' in faculty:
                    message += f"  {faculty['email']}\n"
                message += "\n"
            total = len(math_dept['faculty_members'])
            message += f"*Total Mathematics Faculty: {total} members*"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionFacultySociology(Action):
    def name(self) -> Text:
        return "action_faculty_sociology"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_faculty()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        soc_dept = next((d for d in data.get('departments', [])
                        if 'Sociology' in d.get('department_name', '')), None)
        if soc_dept and 'faculty_members' in soc_dept:
            message = f"**{soc_dept['department_name']} Faculty**\n\n"
            for faculty in soc_dept['faculty_members'][:5]:
                message += f" **{faculty.get('name', 'N/A')}**\n"
                message += f"   {faculty.get('designation', 'N/A')}\n"
                if 'email' in faculty:
                    message += f"  {faculty['email']}\n"
                message += "\n"
            total = len(soc_dept['faculty_members'])
            message += f"*Total Sociology Faculty: {total} members*"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionFacultyInformationStudies(Action):
    def name(self) -> Text:
        return "action_faculty_information_studies"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_faculty()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        is_dept = next((d for d in data.get('departments', [])
                       if 'Information Studies' in d.get('department_name', '')), None)
        if is_dept and 'faculty_members' in is_dept:
            message = f"**{is_dept['department_name']} Faculty**\n\n"
            for faculty in is_dept['faculty_members'][:5]:
                message += f" **{faculty.get('name', 'N/A')}**\n"
                message += f"   {faculty.get('designation', 'N/A')}\n"
                if 'email' in faculty:
                    message += f"  {faculty['email']}\n"
                message += "\n"
            total = len(is_dept['faculty_members'])
            message += f"*Total Information Studies Faculty: {total} members*"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionFacultyPPHS(Action):
    def name(self) -> Text:
        return "action_faculty_pphs"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_faculty()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        pphs_dept = next((d for d in data.get('departments', [])
                         if 'PPHS' in d.get('department_name', '') or 'Public Health' in d.get('department_name', '')), None)
        if pphs_dept and 'faculty_members' in pphs_dept:
            message = f"**{pphs_dept['department_name']} Faculty**\n\n"
            for faculty in pphs_dept['faculty_members'][:5]:
                message += f" **{faculty.get('name', 'N/A')}**\n"
                message += f"   {faculty.get('designation', 'N/A')}\n"
                if 'email' in faculty:
                    message += f"  {faculty['email']}\n"
                message += "\n"
            total = len(pphs_dept['faculty_members'])
            message += f"*Total PPHS Faculty: {total} members*"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionFacultyDataScience(Action):
    def name(self) -> Text:
        return "action_faculty_data_science"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_faculty()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        ds_dept = next((d for d in data.get('departments', [])
                       if 'Data Science' in d.get('department_name', '') or 'Analytics' in d.get('department_name', '')), None)
        if ds_dept and 'faculty_members' in ds_dept:
            message = f"**{ds_dept['department_name']} Faculty**\n\n"
            for faculty in ds_dept['faculty_members'][:5]:
                message += f" **{faculty.get('name', 'N/A')}**\n"
                message += f"   {faculty.get('designation', 'N/A')}\n"
                if 'email' in faculty:
                    message += f"  {faculty['email']}\n"
                message += "\n"
            total = len(ds_dept['faculty_members'])
            message += f"*Total Data Science Faculty: {total} members*"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)

class ActionFacultySocialRelations(Action):
    def name(self) -> Text:
        return "action_faculty_social_relations"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_faculty()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        sr_dept = next((d for d in data.get('departments', [])
                       if 'Social Relations' in d.get('department_name', '')), None)
        if sr_dept and 'faculty_members' in sr_dept:
            message = f"**{sr_dept['department_name']} Faculty**\n\n"
            for faculty in sr_dept['faculty_members'][:5]:
                message += f" **{faculty.get('name', 'N/A')}**\n"
                message += f"   {faculty.get('designation', 'N/A')}\n"
                if 'email' in faculty:
                    message += f"  {faculty['email']}\n"
                message += "\n"
            total = len(sr_dept['faculty_members'])
            message += f"*Total Social Relations Faculty: {total} members*"
            dispatcher.utter_message(text=message)
            return []
        return call_rag_fallback(dispatcher, tracker, domain)
# ========================================
# CHAIRPERSON INFORMATION ACTION
# ========================================

class ActionChairpersonInfo(Action):
    def name(self) -> Text:
        return "action_chairperson_info"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_faculty()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        # Find all chairpersons (those with "Chairperson" in designation)
        chairpersons = [f for f in data.get('faculty', []) 
                       if 'Chairperson' in f.get('designation', '')]
        
        if chairpersons:
            message = "**Department Chairpersons at EWU**\n\n"
            for chair in chairpersons:
                message += f" **{chair.get('name', 'N/A')}**\n"
                message += f"   Department: {chair.get('department', 'N/A')}\n"
                message += f"   Position: {chair.get('designation', 'N/A')}\n"
                if 'profile_url' in chair:
                    message += f"  Profile: {chair['profile_url']}\n"
                message += "\n"
            dispatcher.utter_message(text=message)
            return []
        else:
            dispatcher.utter_message(text="Chairperson information not available.")
            return []
        return call_rag_fallback(dispatcher, tracker, domain)


# ========================================
# GRADING SYSTEM ACTION
# ========================================

class ActionGradingSystem(Action):
    def name(self) -> Text:
        return "action_grading_system"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        data = load_grading()
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        grading = data['grading_system']
        message = f"**{grading['title']}**\n\n"
        message += f"{grading['description']}\n\n"
        message += "**Grade Scale:**\n"
        for grade in grading['grade_scale']:
            message += f"- **{grade['letter_grade']}**: {grade['numerical_score']} - {grade['grade_point']} GPA\n"
        
        message += "\n**Special Grades:**\n"
        for spec_grade in grading['special_grades']:
            message += f"- **{spec_grade['grade']}**: {spec_grade['description']}\n"
        
        dispatcher.utter_message(text=message)
        return []

















# ============================================================================
# DEPARTMENT TO FILE MAPPING
# ============================================================================

DEPARTMENT_FILES = {
    # Undergraduate
    "cse": "st_cse.json",
    "computer science": "st_cse.json",
    "cs": "st_cse.json",
    
    "eee": "st_eee.json",
    "electrical": "st_eee.json",
    
    "ece": "st_ece.json",
    "electronics": "st_ece.json",
    
    "ce": "st_ce.json",
    "civil": "st_ce.json",
    
    "bba": "st_ba.json",
    "business": "st_ba.json",
    
    "economics": "st_economics.json",
    "eco": "st_economics.json",
    
    "english": "st_english.json",
    "eng": "st_english.json",
    
    "pharmacy": "st_pharmacy.json",
    "pharm": "st_pharmacy.json",
    
    "law": "st_law.json",
    
    "math": "st_math.json",
    "mathematics": "st_math.json",
    
    "sociology": "st_sociology.json",
    "soc": "st_sociology.json",
    
    "social relations": "st_social_relations.json",
    
    # Graduate
    "mba": "mba_emba.json",
    "emba": "mba_emba.json",
    
    "ms cse": "ms_cse.json",
    "msc cse": "ms_cse.json",
    
    "ms dsa": "ms_dsa.json",
    "data science": "ms_dsa.json",
    
    "ma english": "ma_english.json",
    
    "mds": "mds.json",
    "development studies": "mds.json",
    
    "mss economics": "mss_eco.json",
    "mss eco": "mss_eco.json",
    
    "mphil pharmacy": "mphil_pharmacy.json",
    
    "tesol": "tesol.json",
}


def normalize_department_name(dept_name: str) -> str:
    """Normalize department name to match DEPARTMENT_FILES keys"""
    
    if not dept_name:
        return ""
    
    # Convert to lowercase and strip
    dept = dept_name.lower().strip()
    
    # Mapping of variations to standard keys
    dept_mappings = {
        # CSE variations
        "computer science and engineering": "cse",
        "computer science & engineering": "cse",
        "computer science": "cse",
        "comp sci": "cse",
        "cs": "cse",
        "cse": "cse",
        
        # EEE variations
        "electrical and electronics engineering": "eee",
        "electrical & electronics engineering": "eee",
        "electrical and electronics": "eee",
        "electrical engineering": "eee",
        "electrical": "eee",
        "eee": "eee",
        "electrical and electronic engineering": "eee",
        "electrical electronic engineering": "eee",     

        
        # ECE variations
        "electronics and communication engineering": "ece",
        "electronics & communication engineering": "ece",
        "electronics and communication": "ece",
        "electronics": "ece",
        "ece": "ece",
        
        # Civil variations
        "civil engineering": "civil",
        "civil": "civil",
        "ce": "civil",
        
        # BBA variations
        "business administration": "bba",
        "business": "bba",
        "bba": "bba",
        "ba": "bba",
        
        # Economics
        "economics": "economics",
        "eco": "economics",
        
        # English
        "english": "english",
        "eng": "english",
        
        # Pharmacy
        "pharmacy": "pharmacy",
        "pharm": "pharmacy",
        
        # Law
        "law": "law",
        
        # Math
        "mathematics": "math",
        "math": "math",
        
        # Sociology
        "sociology": "sociology",
        "soc": "sociology",
        
        # Graduate
        "mba": "mba",
        "emba": "emba",
        "ms cse": "ms cse",
        "msc cse": "ms cse",
        "master of science in cse": "ms cse",
        "ms data science": "ms dsa",
        "data science": "ms dsa",
    }
    
    # Return mapped value or original lowercase
    return dept_mappings.get(dept, dept)







        
#         # Try RAG for unrecognized intents
#         rag_action = ActionPhi3RagAnswer()
#         return rag_action.run(dispatcher, tracker, domain)









# ============================================================================
# ============================================================================

DEPARTMENT_FILES = {
    # Undergraduate
    "cse": "st_cse.json",
    "computer science": "st_cse.json",
    "cs": "st_cse.json",
    
    "eee": "st_eee.json",
    "electrical": "st_eee.json",
    
    "ece": "st_ece.json",
    "electronics": "st_ece.json",
    
    "ce": "st_ce.json",
    "civil": "st_ce.json",
    
    "bba": "st_ba.json",
    "business": "st_ba.json",
    
    "economics": "st_economics.json",
    "eco": "st_economics.json",
    
    "english": "st_english.json",
    "eng": "st_english.json",
    
    "pharmacy": "st_pharmacy.json",
    "pharm": "st_pharmacy.json",
    
    "law": "st_law.json",
    
    "math": "st_math.json",
    "mathematics": "st_math.json",
    
    "sociology": "st_sociology.json",
    "soc": "st_sociology.json",
    
    "social relations": "st_social_relations.json",
    
    # Graduate
    "mba": "mba_emba.json",
    "emba": "mba_emba.json",
    
    "ms cse": "ms_cse.json",
    "msc cse": "ms_cse.json",
    
    "ms dsa": "ms_dsa.json",
    "data science": "ms_dsa.json",
    
    "ma english": "ma_english.json",
    
    "mds": "mds.json",
    "development studies": "mds.json",
    
    "mss economics": "mss_eco.json",
    "mss eco": "mss_eco.json",
    
    "mphil pharmacy": "mphil_pharmacy.json",
    
    "tesol": "tesol.json",
}


def normalize_department_name(dept_name: str) -> str:
    """Normalize department name to match DEPARTMENT_FILES keys"""
    
    if not dept_name:
        return ""
    
    # Convert to lowercase and strip
    dept = dept_name.lower().strip()
    
    # Mapping of variations to standard keys
    dept_mappings = {
        # CSE variations
        "computer science and engineering": "cse",
        "computer science & engineering": "cse",
        "computer science": "cse",
        "comp sci": "cse",
        "cs": "cse",
        "cse": "cse",
        
        # EEE variations
        "electrical and electronics engineering": "eee",
        "electrical & electronics engineering": "eee",
        "electrical and electronics": "eee",
        "electrical engineering": "eee",
        "electrical": "eee",
        "electrical and electronic engineering" 
        "eee": "eee",
        
        # ECE variations
        "electronics and communication engineering": "ece",
        "electronics & communication engineering": "ece",
        "electronics and communication": "ece",
        "electronics": "ece",
        "ece": "ece",
        
        # Civil variations
        "civil engineering": "civil",
        "civil": "civil",
        "ce": "civil",
        
        # BBA variations
        "business administration": "bba",
        "business": "bba",
        "bba": "bba",
        "ba": "bba",
        
        # Economics
        "economics": "economics",
        "eco": "economics",
        
        # English
        "english": "english",
        "eng": "english",
        
        # Pharmacy
        "pharmacy": "pharmacy",
        "pharm": "pharmacy",
        
        # Law
        "law": "law",
        
        # Math
        "mathematics": "math",
        "math": "math",
        
        # Sociology
        "sociology": "sociology",
        "soc": "sociology",
        
        # Graduate
        "mba": "mba",
        "emba": "emba",
        "ms cse": "ms cse",
        "msc cse": "ms cse",
        "master of science in cse": "ms cse",
        "ms data science": "ms dsa",
        "data science": "ms dsa",
    }
    
    # Return mapped value or original lowercase
    return dept_mappings.get(dept, dept)




# ============================================================================
# ACTION: Show All Courses
# ============================================================================

class ActionShowCourses(Action):
    """Show all courses for a department"""
    
    def name(self) -> Text:
        return "action_show_courses"
    
    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:
        
        # Get department from slot or entity
        department = normalize_department_name(tracker.get_slot("department"))
        
        if not department:
            # Try to get from entities
            entities = tracker.latest_message.get('entities', [])
            for entity in entities:
                if entity['entity'] == 'department':
                    department = entity['value']
                    break
        
        if not department:
            dispatcher.utter_message(
                text="Please specify a department. For example: 'CSE courses' or 'BBA courses'"
            )
            return []
        
        # Find matching file
        filename = DEPARTMENT_FILES.get(department)
        
        if not filename:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        # Load course data
        data = load_json_file(filename)
        
        if not data:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        # Extract courses
        courses = self._extract_courses(data)
        
        if not courses:
            dispatcher.utter_message(
                text=f"No courses found for {department}."
            )
            return [SlotSet("department", None)]
        
        # Format and send response
        response = self._format_courses_response(department, courses, data)
        dispatcher.utter_message(text=response)
        
        return [SlotSet("department", department)]
    
    def _extract_courses(self, data: dict) -> List[dict]:
        """Extract course list from JSON data"""
        courses = []
        
        # Try different JSON structures
        if 'courses' in data:
            courses = data['courses']
        elif 'course_list' in data:
            courses = data['course_list']
        elif 'curriculum' in data:
            courses = data['curriculum']
        elif isinstance(data, list):
            courses = data
        
        return courses
    
    def _format_courses_response(self, department: str, courses: List[dict], data: dict) -> str:
        """Format courses into readable response"""
        
        # Get department info
        dept_info = data.get('department_info', {})
        
        # Get program/department name
        program_name = (
            dept_info.get('program_name') or 
            dept_info.get('department_name') or 
            data.get('program_name') or
            department.upper()
        )
        
        # Get total credits
        total_credits = (
            dept_info.get('total_credits') or 
            dept_info.get('minimum_credits_required') or
            data.get('total_credits') or
            'N/A'
        )
        
        response = f"📚 **{program_name} Courses**\n\n"
        response += f"**Total Credits:** {total_credits}\n\n"
        
        # Show course summaries if available
        course_summaries = data.get('course_summaries', {})
        if course_summaries:
            response += "**Credit Distribution:**\n"
            for category, credits in course_summaries.items():
                category_name = category.replace('_', ' ').title()
                response += f"• {category_name}: {credits}\n"
            response += "\n"
        
        # Group courses by category
        grouped = self._group_courses_by_category(courses)
        
        if grouped:
            for category, cat_courses in list(grouped.items())[:5]:
                response += f"**{category}:**\n"
                for course in cat_courses[:8]:
                    code = course.get('code', course.get('course_code', 'N/A'))
                    name = course.get('name', course.get('title', 'N/A'))
                    credits = course.get('credits', course.get('credit', 'N/A'))
                    response += f"• {code} - {name} ({credits} cr)\n"
                
                if len(cat_courses) > 8:
                    response += f"  ... and {len(cat_courses) - 8} more\n"
                response += "\n"
        else:
            response += "**Course List:**\n"
            for course in courses[:15]:
                code = course.get('code', course.get('course_code', 'N/A'))
                name = course.get('name', course.get('title', 'N/A'))
                credits = course.get('credits', course.get('credit', 'N/A'))
                response += f"• {code} - {name} ({credits} cr)\n"
            
            if len(courses) > 15:
                response += f"\n... and {len(courses) - 15} more courses\n\n"
        
        response += f"Please visit https://www.ewubd.edu for more information."
        return response
    
    def _group_courses_by_category(self, courses: List[dict]) -> Dict[str, List[dict]]:
        grouped = {}
        for course in courses:
            category = (
                course.get('category') or 
                course.get('semester') or 
                course.get('year') or 
                course.get('level') or
                'Other Courses'
            )
            if category not in grouped: grouped[category] = []
            grouped[category].append(course)
        return grouped if len(grouped) > 1 else {}

class ActionShowCourseDetails(Action):
    """Show details of a specific course by code"""
    def name(self) -> Text:
        return "action_show_course_details"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        course_code = tracker.get_slot("course_code")
        if not course_code:
            entities = tracker.latest_message.get('entities', [])
            for entity in entities:
                if entity['entity'] == 'course_code':
                    course_code = entity['value']; break
        if not course_code:
            dispatcher.utter_message(text="Please provide a course code. For example: CSE101")
            return []
        
        course_code = course_code.upper().replace(' ', '').replace('-', '')
        course_info = self._find_course(course_code)
        
        if not course_info:
            return call_rag_fallback(dispatcher, tracker, domain)
        
        response = self._format_course_details(course_code, course_info)
        dispatcher.utter_message(text=response)
        return [SlotSet("course_code", course_code)]
    
    def _find_course(self, course_code: str) -> dict:
        for filename in set(DEPARTMENT_FILES.values()):
            data = load_json_file(filename)
            if not data: continue
            courses = data.get('courses', data if isinstance(data, list) else [])
            for course in courses:
                code = course.get('code', course.get('course_code', '')).upper().replace(' ', '').replace('-', '')
                if code == course_code: return course
        return None
    
    def _format_course_details(self, course_code: str, course: dict) -> str:
        code = course.get('code', course.get('course_code', course_code))
        name = course.get('name', course.get('title', 'N/A'))
        credits = course.get('credits', course.get('credit', 'N/A'))
        prereq = course.get('prerequisites', course.get('prerequisite', 'None'))
        category = course.get('category', 'N/A')
        desc = course.get('description', 'No description available')
        
        res = f"**Course Details**\n\n**Code:** {code}\n**Name:** {name}\n**Credits:** {credits}\n**Prerequisites:** {prereq}\n**Category:** {category}\n\n"
        if desc and desc != 'No description available': res += f"**Description:**\n{desc}\n\n"
        res += "Please visit https://www.ewubd.edu for more information."
        return res

class ActionShowCredits(Action):
    """Show total credits for a program"""
    def name(self) -> Text:
        return "action_show_credits"
    
    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        department = normalize_department_name(tracker.get_slot("department"))
        if not department:
            entities = tracker.latest_message.get('entities', [])
            for entity in entities:
                if entity['entity'] == 'department':
                    department = entity['value']; break
        if not department:
            dispatcher.utter_message(text="Which program? (e.g., CSE, BBA)")
            return []
        
        filename = DEPARTMENT_FILES.get(department)
        if not filename: return call_rag_fallback(dispatcher, tracker, domain)
        data = load_json_file(filename)
        if not data: return call_rag_fallback(dispatcher, tracker, domain)
        
        dept_info = data.get('department_info', {})
        program_name = dept_info.get('program_name') or dept_info.get('department_name') or data.get('program_name') or department.upper()
        total_credits = dept_info.get('total_credits') or dept_info.get('minimum_credits_required') or data.get('total_credits') or 'N/A'
        
        res = f"**{program_name} Credit Requirements**\n\n**Total Credits:** {total_credits}\n\n"
        breakdown = data.get('course_summaries', data.get('credit_breakdown', {}))
        if breakdown:
            res += "**Credit Breakdown:**\n"
            for cat, creds in breakdown.items():
                res += f"• {cat.replace('_', ' ').title()}: {creds}\n"
        res += "\nPlease visit https://www.ewubd.edu for more information."
        dispatcher.utter_message(text=res)
        return [SlotSet("department", department)]
