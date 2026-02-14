import math

def calculate_confidence(scores):
    if scores:
        return 1 / (1 + math.exp(-max(scores)))
    return 0.5

def build_prompt(context_text, query):
    return (
        "<s>[INST] You are an assistant for East West University (EWU) in Bangladesh.\n"
        "Answer the question using ONLY the context below. Be concise and helpful.\n"
        "If the context doesn't contain the answer, say \"I don't have that information.\"\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {query} [/INST]"
    )

def test_logic():
    # Test confidence calculation
    assert round(calculate_confidence([0]), 3) == 0.500
    assert calculate_confidence([2]) > 0.5
    assert calculate_confidence([-2]) < 0.5
    print("Confidence logic passed.")

    # Test prompt building
    context = "Context 1\n\nContext 2"
    query = "What is EWU?"
    prompt = build_prompt(context, query)
    assert "[INST]" in prompt
    assert "[/INST]" in prompt
    assert "Context 1" in prompt
    assert query in prompt
    print("Prompt logic passed.")

if __name__ == "__main__":
    test_logic()
