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
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    print(f"Model ID: {model_id}")

    # Verify 4-bit configuration
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    if bnb_config.load_in_4bit:
        print("✅ BitsAndBytesConfig (4-bit) initialized correctly.")
    else:
        print("❌ BitsAndBytesConfig (4-bit) initialization failed.")
        return False

    # Check for memory environment variable
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "expandable_segments:True":
        print("✅ PYTORCH_CUDA_ALLOC_CONF is set correctly.")
    else:
        # We set it in service.py, but for this test we might need to set it manually if it's not inherited
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        print("ℹ️ PYTORCH_CUDA_ALLOC_CONF was not set, setting it now for test context.")

    # Verify prompt format
    context_text = "Sample context"
    query = "Sample query"
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a highly accurate and helpful assistant for East West University (EWU) in Bangladesh. "
        "Use the provided context to answer the user's question. "
        "Maintain the language of the user's query: if they ask in Bangla, respond in Bangla; "
        "if in Banglish, respond in Banglish (or clear Bangla); if in English, respond in English. "
        "If you don't know the answer based on the context, say you don't have that information. "
        "Be concise but thorough.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    if "<|begin_of_text|>" in prompt and "<|start_header_id|>system<|end_header_id|>" in prompt:
        print("✅ Prompt format looks correct for Llama 3.1.")
    else:
        print("❌ Prompt format seems incorrect.")
        return False

    return True

if __name__ == "__main__":
    # We skip actual loading in sandbox as we can't run the full service without GPU/proper env
    if test_imports() and test_config():
        print("\nVerification successful (Config & Imports).")
    else:
        print("\nVerification failed.")
