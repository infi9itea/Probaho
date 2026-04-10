#!/bin/bash

# Start Rasa Action Server
cd rasa
rasa run actions --port 5055 &

# Start Rasa Server
rasa run --enable-api --cors "*" --port 5005 &

# Wait for services to initialize
sleep 20

# Start FastAPI backend
cd ../backend
uvicorn main:app --host 0.0.0.0 --port 7860

# Keep script running
wait -n
exit $?
