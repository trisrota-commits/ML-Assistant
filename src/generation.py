import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.to("cpu")
model.eval()

MODE_PROMPTS = {
    "concise": "Answer in 2-3 short sentences.",
    "detailed": "Provide a thorough technical explanation.",
}


def generate_answer(
    question: str,
    context: str,
    mode: str = "concise",
    grounded: bool = True,
):
    system_prompt = MODE_PROMPTS[mode]
    if grounded:
        system_prompt += " Answer only using the provided context. If insufficient, say so."
    else:
        system_prompt += (
            " No retrieved context was available. Answer from general knowledge, "
            "and clearly state that the answer is not grounded in the document collection."
        )

    prompt = f"""<|system|>
{system_prompt}
</s>
<|user|>
Context:
{context}

Question: {question}
</s>
<|assistant|>
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1800
    ).to("cpu")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def rewrite_query(question: str) -> str:
    """Rewrite a query toward terminology likely to occur in the source blogs."""
    prompt = (
        "Rewrite the following information-retrieval query using precise machine-learning "
        "terminology. Return only the rewritten query.\n\nQuery: " + question
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to("cpu")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()