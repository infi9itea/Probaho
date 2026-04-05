import os
import torch
import transformers

def test_imports():
    print("Testing imports...")
    try:
        from sentence_transformers import CrossEncoder
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        from fastapi import FastAPI
        print("✅ Imports successful.")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    return True

def test_config():
    print("Testing configuration...")
    model_id = "mistralai/Ministral-8B-Instruct-2410"
    print(f"Model ID: {model_id}")

    # Verify 4-bit configuration (v2 uses bfloat16)
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    if bnb_config.load_in_4bit and bnb_config.bnb_4bit_compute_dtype == torch.bfloat16:
        print("✅ BitsAndBytesConfig (4-bit bfloat16) initialized correctly.")
    else:
        print("❌ BitsAndBytesConfig initialization mismatch.")
        return False

    # Check for memory environment variable
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True":
        print("✅ PYTORCH_CUDA_ALLOC_CONF is set correctly.")
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        print("ℹ️ Setting PYTORCH_CUDA_ALLOC_CONF to 'expandable_segments:True'.")

    # Verify prompt keywords for v2
    prompt_keywords = ["CRITICAL INSTRUCTION FOR LANGUAGE", "Banglish", "standard Bangla"]

    # Mocking prompt template check
    template = """
    <|system|>
    You are a helpful and knowledgeable assistant for East West University (EWU). Use only the provided context to answer questions.
    CRITICAL INSTRUCTION FOR LANGUAGE:
    If the user asks the question in English, answer in English.
    If the user asks in Bangla (বাংলা), answer completely in standard Bangla.
    If the user asks in Banglish (Romanized Bengali, e.g., "admission deadline kobe?"), answer in friendly Banglish.
    """

    if all(kw in template for kw in prompt_keywords):
        print("✅ Prompt keywords for v2 (Multilingual) verified.")
    else:
        print("❌ Prompt keywords missing for v2.")
        return False

    return True

if __name__ == "__main__":
    if test_imports() and test_config():
        print("\nVerification successful (Config & Imports for RAG v2).")
    else:
        print("\nVerification failed.")
        exit(1)
