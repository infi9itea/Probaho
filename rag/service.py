import os
import time
import math
from fastapi import FastAPI
from pydantic import BaseModel
from retriever import Retriever
import transformers
import torch
import bitsandbytes as bnb
from huggingface_hub import login

HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)

VECTORSTORE_PATH = "vectorstore"
retriever = Retriever(VECTORSTORE_PATH)

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    token=HF_TOKEN,
    quantization_config=bnb_config,
    device_map="auto"
)

generator = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    return_full_text=False,
    do_sample=False
)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 20

app = FastAPI()

@app.post("/rag/query")
def rag_query(req: QueryRequest):
    start = time.time()

    contexts = retriever.retrieve(req.query, top_k=req.top_k, return_k=20)
    if not contexts:
        return {
            "response": "I don't have that information in my database.",
            "confidence": 0.0,
            "sources": [],
            "processing_time": round(time.time() - start, 3)
        }

    selected = contexts[:5]
    context_text = "\n\n".join([c["text"] for c in selected])

    prompt = (
        "<s>[INST] You are an assistant for East West University (EWU) in Bangladesh.\n"
        "Answer the question using ONLY the context below. Be concise and helpful.\n"
        "If the context doesn't contain the answer, say \"I don't have that information.\"\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {req.query} [/INST]"
    )

    try:
        generated = generator(prompt)[0]["generated_text"].strip()
    except Exception as e:
        return {
            "response": f"Error generating response: {str(e)}",
            "confidence": 0.0,
            "sources": [c.get("source", "") for c in selected],
            "processing_time": round(time.time() - start, 3)
        }

    scores = [c.get("score", 0.0) for c in selected]
    if scores:
        # Map reranker logit to [0, 1] range using sigmoid
        confidence = 1 / (1 + math.exp(-max(scores)))
    else:
        confidence = 0.5

    return {
        "response": generated,
        "confidence": round(confidence, 3),
        "sources": [c.get("source", "") for c in selected],
        "processing_time": round(time.time() - start, 3)
    }