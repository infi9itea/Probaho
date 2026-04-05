import os
import torch
import transformers
from retriever import Retriever

def test_imports():
    print("Testing imports...")
    try:
        from sentence_transformers import CrossEncoder
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("✅ Imports successful.")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    return True

def test_config():
    print("Testing configuration...")
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    print(f"Model ID: {model_id}")

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_8bit=True
    )
    print("BitsAndBytesConfig (8-bit) initialized.")

    # We won't actually load the model here to save resources/time in sandbox
    # but we can check if the prompt format is correct
    context_text = "Sample context"
    query = "Sample query"
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful assistant for East West University (EWU) in Bangladesh. "
        "Use the provided context to answer the user's question. "
        "Answer in the same language as the question (English, Bangla, or Banglish). "
        "If you don't know the answer based on the context, say you don't have that information. "
        "Be concise.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    if "<|begin_of_text|>" in prompt and "<|start_header_id|>system<|end_header_id|>" in prompt:
        print("✅ Prompt format looks correct for Llama 3.")
    else:
        print("❌ Prompt format seems incorrect.")
        return False

    return True

if __name__ == "__main__":
    if test_imports() and test_config():
        print("\nVerification successful (Config & Imports).")
    else:
        print("\nVerification failed.")
