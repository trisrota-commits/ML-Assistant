import gradio as gr
from src.rag_core import query_rag

def ask(question):
    answer, results = query_rag(question)
    return answer

demo = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(label="Ask a question"),
    outputs=gr.Textbox(label="Answer"),
    title="ML RAG Assistant",
    description="Retrieval-Augmented QA over ML blogs (D2L, Karpathy, Ruder, Colah)"
)

if __name__ == "__main__":
    demo.launch()