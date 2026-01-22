import os
from fastapi import FastAPI
from pydantic import BaseModel
from retriever import Retriever
import transformers
import torch
import bitsandbytes as bnb
from huggingface_hub import login

# Authenticate
HF_TOKEN = os.getenv("HF_TOKEN")
login(HF_TOKEN)

VECTORSTORE_PATH = "vectorstore"
retriever = Retriever(VECTORSTORE_PATH)

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# Quantization config
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
    max_new_tokens=256,
    return_full_text=False,  # Only return the answer
    do_sample=False
)

class QueryRequest(BaseModel):
    query: str

app = FastAPI()

@app.post("/rag/query")
def rag_query(req: QueryRequest):
    contexts = retriever.retrieve(req.query)
    if not contexts:
        return {
            "answer": "I don't have that information in my database.",
            "confidence": 0.0
        }

    context_text = "\n\n".join(contexts)

    prompt = f"""<s>[INST] You are a university information assistant. Use ONLY the context below to answer the question. If the answer is not in the context, say "I don't have that information."

Context:
{context_text}

Question: {req.query} [/INST]"""

    try:
        answer = generator(prompt)[0]["generated_text"].strip()
    except Exception as e:
        answer = f"Error generating response: {str(e)}"

    return {
        "answer": answer,
        "confidence": 0.8
    }