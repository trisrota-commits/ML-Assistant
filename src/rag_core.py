from src.data_loader import fetch_clean_text, clean_text
from src.chunking import chunk_text
from src.retrieval import Retriever
from src.generation import generate_answer


def build_documents():
    blog_urls = [
        "http://karpathy.github.io/2019/04/25/recipe/",
        "https://ruder.io/optimizing-gradient-descent/",
        "https://colah.github.io/posts/2015-08-Understanding-LSTMs/"
    ]

    documents = []

    for url in blog_urls:
        text = fetch_clean_text(url)
        text = clean_text(text)
        chunks = chunk_text(text)

        if "karpathy" in url:
            label = "Karpathy"
        elif "ruder" in url:
            label = "Ruder"
        else:
            label = "Colah"

        for chunk in chunks:
            documents.append({"text": chunk, "source": label})

    return documents


documents = build_documents()
retriever = Retriever(documents)


def query_rag(question: str, mode: str = "concise"):
    results = retriever.retrieve(question, k=2)
    context = "\n".join([r["text"] for r in results])
    answer = generate_answer(question, context, mode)
    if not results:
        return "The retrieved context does not contain relevant information.", []
    return answer, results