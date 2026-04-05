import os

# Set environment variable for memory management before other imports
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import math
from fastapi import FastAPI
from pydantic import BaseModel
from retriever import Retriever
import transformers
import torch
from huggingface_hub import login

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable must be set.")
login(HF_TOKEN)

VECTORSTORE_PATH = "vectorstore"
retriever = Retriever(VECTORSTORE_PATH)

model_id = "mistralai/Ministral-8B-Instruct-2410"

# Memory optimization: 4-bit quantization as per v2
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    token=HF_TOKEN,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

generator = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=768,      # Updated to v2
    temperature=0.3,         # Updated to v2
    top_p=0.9,               # Updated to v2
    repetition_penalty=1.05,  # Updated to v2
    return_full_text=False,
    do_sample=True
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 25  # Updated to v2

app = FastAPI()

@app.post("/rag/query")
def rag_query(req: QueryRequest):
    start = time.time()

    # Retrieve and rerank using updated defaults
    contexts = retriever.retrieve(req.query, top_k=req.top_k, return_k=8)
    if not contexts:
        return {
            "response": "I don't have enough information to answer that. / আমার কাছে এই তথ্যটি নেই।",
            "confidence": 0.0,
            "sources": [],
            "processing_time": round(time.time() - start, 3)
        }

    # As per snippet, use reranked contexts (up to 8)
    context_text = "\n\n".join([c["text"] for c in contexts])

    # Multilingual Prompt v2 instructions
    system_instruction = (
        "You are a helpful and knowledgeable assistant for East West University (EWU). Use only the provided context to answer questions.\n\n"
        "CRITICAL INSTRUCTION FOR LANGUAGE:\n"
        "If the user asks the question in English, answer in English. "
        "If the user asks in Bangla (বাংলা), answer completely in standard Bangla. "
        "If the user asks in Banglish (Romanized Bengali, e.g., 'admission deadline kobe?'), answer in friendly Banglish. "
        "If unsure, say: \"I don't have enough information to answer that.\" / \"আমার কাছে এই তথ্যটি নেই।\" "
        "Keep answers accurate and concise."
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"CONTEXT:\n{context_text}\n\nQUESTION: {req.query}"}
    ]

    try:
        # Mistral v2 prompt format as per snippet (translated into chat template or similar)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        generated = generator(prompt)[0]["generated_text"].strip()
    except Exception as e:
        return {
            "response": f"Error generating response: {str(e)}",
            "confidence": 0.0,
            "sources": [c.get("source", "") for c in contexts],
            "processing_time": round(time.time() - start, 3)
        }

    scores = [c.get("score", 0.0) for c in contexts]
    if scores:
        confidence = 1 / (1 + math.exp(-max(scores)))
    else:
        confidence = 0.5

    return {
        "response": generated,
        "confidence": round(confidence, 3),
        "sources": [c.get("source", "") for c in contexts],
        "processing_time": round(time.time() - start, 3)
    }
