# Probaho - University Information Guide

Probaho is an information guide for East West University, powered by Rasa and a RAG (Retrieval-Augmented Generation) system. This repository is configured for deployment on Hugging Face Spaces.

## How to Run

### 1. Local Deployment (Docker)

To run the entire system locally using Docker:

```bash
docker build -t probaho .
docker run -p 7860:7860 probaho
```

The application will be available at `http://localhost:7860`.

### 2. Hugging Face Spaces Deployment

1. Create a new **Docker** Space on Hugging Face.
2. Push this repository to your Space.
3. Ensure the following environment variable is set in your Space settings:
   - `RAG_API_URL`: The URL of your remote RAG service.

The Space will automatically build using the `Dockerfile` and start the services on port 7860.

## Project Structure

- `rasa/`: Rasa configuration and training data.
- `rasa/actions/`: Custom action server logic (RAG-powered).
- `backend/`: FastAPI proxy and static file server.
- `rag/`: RAG pipeline and ingestion scripts.
- `static/`: Bundled frontend assets (index.html, app.js, style.css).

## Features

- **Multilingual Support**: Supports English, Bangla, and Banglish (Romanized Bengali).
- **RAG-Powered**: Real-time information retrieval from university data.
- **Unified Architecture**: All components (Rasa, Actions, Backend) run in a single container.
- **Persistent Sessions**: Chat history is maintained during the user session.

## Credits

RAG data was contributed by Mohua, Samir, and the development team. The RAG pipeline was designed by the core team, and the Rasa pipeline and data were contributed by Atkiya Maisha.
