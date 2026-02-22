from transformers import AutoTokenizer

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def chunk_text(text: str, chunk_words: int = 400, overlap: int = 100):
    chunks = []
    words = text.split()

    start = 0
    while start < len(words):
        chunk = " ".join(words[start:start + chunk_words])
        token_ids = tokenizer(chunk, add_special_tokens=False)["input_ids"]
        chunks.append(tokenizer.decode(token_ids))
        start += chunk_words - overlap

    return chunks