from fastapi import FastAPI
from pydantic import BaseModel
from retriever import Retriever
import transformers
import torch

VECTORSTORE_PATH = "vectorstore"

retriever = Retriever(VECTORSTORE_PATH)

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    device_map="auto"
)

generator = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.0,
    max_new_tokens=256
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

    prompt = f"""
You are a university information assistant.
Use only the context below.

Context:
{context_text}

Question:
{req.query}

Answer:
"""

    output = generator(prompt)[0]["generated_text"]
    answer = output.split("Answer:")[-1].strip()

    return {
        "answer": answer,
        "confidence": 0.8
    }
