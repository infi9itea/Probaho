from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rasa_client import send_to_rasa
import requests
import os
import time
from typing import List, Optional, Dict, Any

app = FastAPI()

# Allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DATA_URL = "https://raw.githubusercontent.com/infi9itea/ewu_data/main/"
RAG_API_URL = os.environ.get("RAG_API_URL")

# Simple in-memory cache for remote JSON data
DATA_CACHE = {}
CACHE_TTL = 3600  # 1 hour

def fetch_data(filename: str):
    now = time.time()
    if filename in DATA_CACHE:
        cache_entry, timestamp = DATA_CACHE[filename]
        if now - timestamp < CACHE_TTL:
            return cache_entry

    url = f"{BASE_DATA_URL}{filename}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            DATA_CACHE[filename] = (data, now)
            return data
        return None
    except Exception:
        return None

def call_rag(query: str):
    if not RAG_API_URL:
        return {"response": "RAG service URL not configured.", "confidence": 0.0}

    headers = {"Ngrok-Skip-Browser-Warning": "true"}
    payload = {"query": query}
    try:
        response = requests.post(RAG_API_URL, json=payload, headers=headers, timeout=50)
        if response.status_code == 200:
            return response.json()
        return {"response": "Error connecting to RAG service.", "confidence": 0.0}
    except Exception as e:
        return {"response": f"RAG error: {str(e)}", "confidence": 0.0}

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/chat")
async def chat(request: ChatRequest):
    response = send_to_rasa(request.session_id, request.message)
    return response

# Academic Endpoints
@app.get("/api/departments")
async def list_departments():
    data = fetch_data("static_depts.json")
    if data: return data
    raise HTTPException(status_code=404, detail="Departments data not found")

@app.get("/api/programs")
async def list_programs():
    data = fetch_data("static_AllAvailablePrograms.json")
    if data: return data
    raise HTTPException(status_code=404, detail="Programs data not found")

@app.get("/api/programs/{program_id}")
async def get_program(program_id: str):
    data = fetch_data("static_AllAvailablePrograms.json")
    if data:
        target = program_id.lower()
        if isinstance(data, list):
            for p in data:
                if str(p.get('id', '')).lower() == target or str(p.get('code', '')).lower() == target or str(p.get('program_name', '')).lower() == target:
                    return p
        elif isinstance(data, dict):
            for key in ['undergraduate', 'graduate', 'diploma']:
                if key in data:
                    for p in data[key]:
                        if str(p.get('id', '')).lower() == target or str(p.get('code', '')).lower() == target or str(p.get('program_name', '')).lower() == target:
                            return p
    return call_rag(f"Tell me about program {program_id} at East West University")

@app.get("/api/grade-scale")
async def list_grade_scale():
    data = fetch_data("dynamic_grading.json")
    if data: return data
    raise HTTPException(status_code=404, detail="Grade scale data not found")

@app.get("/api/admission-deadlines")
async def list_admission_deadlines():
    data = fetch_data("dynamic_admission_calendar.json")
    if data: return data
    raise HTTPException(status_code=404, detail="Admission deadlines data not found")

@app.get("/api/academic-calendar")
async def list_academic_calendar():
    data = fetch_data("dynamic_admission_calendar.json")
    if data: return data
    raise HTTPException(status_code=404, detail="Academic calendar data not found")

# People Endpoints
@app.get("/api/faculty")
async def list_faculty():
    data = fetch_data("dynamic_faculty.json")
    if data: return data
    return call_rag("List all faculty members of East West University")

@app.get("/api/faculty/{faculty_id}")
async def get_faculty_member(faculty_id: str):
    data = fetch_data("dynamic_faculty.json")
    target = faculty_id.lower()
    if data and 'departments' in data:
        for dept in data['departments']:
            for member in dept.get('faculty_members', []):
                if str(member.get('id', '')).lower() == target or member.get('name', '').lower() == target:
                    return member
    return call_rag(f"Tell me about faculty member {faculty_id}")

@app.get("/api/governance")
async def list_governance():
    data = fetch_data("static_Admin.json")
    if data: return data
    raise HTTPException(status_code=404, detail="Governance data not found")

@app.get("/api/alumni")
async def list_alumni():
    data = fetch_data("static_alumni.json")
    if data: return data
    return call_rag("List notable alumni of East West University")

# Campus Endpoints
@app.get("/api/clubs")
async def list_clubs():
    data = fetch_data("static_clubs.json")
    if data: return data
    raise HTTPException(status_code=404, detail="Clubs data not found")

@app.get("/api/events")
async def list_events():
    data = fetch_data("dynamic_events_workshops.json")
    if data: return data
    return call_rag("What are the upcoming events at East West University?")

@app.get("/api/notices")
async def list_notices():
    data = fetch_data("static_notices.json")
    if data: return data
    return call_rag("What are the latest notices at East West University?")

@app.get("/api/helpdesk")
async def list_helpdesk():
    data = fetch_data("static_helpdesk.json")
    if data: return data
    raise HTTPException(status_code=404, detail="Helpdesk data not found")

@app.get("/api/proctor-schedule")
async def list_proctor_schedule():
    data = fetch_data("static_proctor_schedule.json")
    if data: return data
    return call_rag("What is the proctor schedule at East West University?")

# Finance Endpoints
@app.get("/api/tuition-fees")
async def list_tuition_fees():
    data = fetch_data("dynamic_tution_fees.json")
    if data: return data
    raise HTTPException(status_code=404, detail="Tuition fees data not found")

@app.get("/api/scholarships")
async def list_scholarships():
    data = fetch_data("static_scholarship_and_financial.json")
    if data: return data
    raise HTTPException(status_code=404, detail="Scholarships data not found")

# Information Endpoints
@app.get("/api/documents")
async def list_documents():
    data = fetch_data("static_documents.json")
    if data: return data
    return call_rag("What documents are available for East West University students?")

@app.get("/api/documents/{slug}")
async def get_document(slug: str):
    data = fetch_data("static_documents.json")
    if data and isinstance(data, list):
        doc = next((d for d in data if d.get('slug') == slug), None)
        if doc: return doc
    return call_rag(f"Tell me about the document with slug {slug}")

@app.get("/api/policies")
async def list_policies():
    data = fetch_data("static_Policy.json")
    if data: return data
    raise HTTPException(status_code=404, detail="Policies data not found")

@app.get("/api/newsletters")
async def list_newsletters():
    data = fetch_data("static_newsletters.json")
    if data: return data
    return call_rag("Are there any newsletters available for East West University?")

@app.get("/api/partnerships")
async def list_partnerships():
    data = fetch_data("static_partnerships.json")
    if data: return data
    return call_rag("What are the partnerships of East West University?")

# Search Endpoint
@app.get("/api/search")
async def search(q: str = Query(...)):
    return call_rag(q)

# Courses Endpoints
@app.get("/api/courses/programs")
async def list_course_programs():
    data = fetch_data("static_AllAvailablePrograms.json")
    if data: return data
    raise HTTPException(status_code=404, detail="Course programs data not found")

@app.get("/api/courses/programs/{program_code}")
async def get_course_program(program_code: str):
    filename = f"st_{program_code.lower()}.json"
    data = fetch_data(filename)
    if data: return data
    return call_rag(f"Tell me about the {program_code} program courses at East West University")

@app.get("/api/courses")
async def list_courses():
    return call_rag("List all course offerings at East West University")

@app.get("/api/courses/{course_code}")
async def get_course_offering(course_code: str):
    return call_rag(f"Tell me about course {course_code} at East West University")
