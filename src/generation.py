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


def generate_answer(question: str, context: str, mode: str = "concise"):
    system_prompt = MODE_PROMPTS[mode]
    system_prompt += " Answer only using the provided context. If insufficient, say so."

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