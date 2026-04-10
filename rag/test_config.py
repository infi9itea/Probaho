import torch
import transformers
import bitsandbytes as bnb
import os

def test_config():
    print("Testing RAG v2 configuration...")

    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Count: {torch.cuda.device_count()}")

    try:
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        print("✅ BitsAndBytes 4-bit config created successfully.")
    except Exception as e:
        print(f" BitsAndBytes config error: {e}")

    print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")

    print("\nConfiguration check complete.")

if __name__ == "__main__":
    test_config()
