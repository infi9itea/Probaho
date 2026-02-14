# Probaho

Probaho is a comprehensive university information chatbot system for East West University. It integrates Rasa for intent-based interactions and a Retrieval-Augmented Generation (RAG) service for open-ended questions based on university documents.

## Project Structure

- `backend/`: FastAPI backend that serves as a bridge between the frontend and the Rasa service.
- `rasa/`: Rasa chatbot configuration, including NLU data, stories, and custom actions in `rasa/actions`.
- `rag/`: RAG service using Mistral-7B and FAISS for document retrieval and question answering.
- `frontend`: Simple HTML/JS/CSS frontend.

## Prerequisites

- Docker and Docker Compose
- Hugging Face Token (for the RAG service)

## Setup and Running

1. **Clone the repository**
2. **Set environment variables**
   Create a `.env` file in the root directory and add your Hugging Face token:
   ```env
   HF_TOKEN=your_hugging_face_token_here
   ```
3. **Run with Docker Compose**
   ```bash
   docker-compose up --build
   ```
   This will start all services:
   - Rasa: `http://localhost:5005`
   - Rasa Actions: `http://localhost:5055`
   - Backend: `http://localhost:8000`
   - RAG: `http://localhost:8001`

4. **Access the Chat Interface**
   Open `index.html` in your browser. The frontend is configured to communicate with the backend at `http://localhost:8000/chat`.

## Key Features

- **Intent-based Responses**: Handled by Rasa for common university queries (admissions, tuition, departments, etc.).
- **RAG-powered Q&A**: Uses Mistral-7B to answer questions based on the university's knowledge base when specific intents aren't matched.
- **Microservices Architecture**: Containerized services for easy deployment and scaling.
