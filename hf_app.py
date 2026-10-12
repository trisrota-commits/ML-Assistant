import os

# Enable ZeroGPU emulation before any dependency imports torch.
if os.environ.get("RAG_DEVICE", "cpu") == "cuda":
    import spaces

import gradio as gr
from src.agent import agentic_query
# Load CUDA weights before ZeroGPU packs tensors; defer CPU encoding to its worker.
from src.generation import model

def ask(question):
    answer, results, metadata = agentic_query(question)
    return answer, {
        **metadata,
        "retrieved_ids": [int(result["id"]) for result in results],
    }

if os.environ.get("RAG_DEVICE", "cpu") == "cuda":
    ask = spaces.GPU(ask)

demo = gr.Interface(
    fn=ask,
    inputs=gr.Textbox(label="Ask a question"),
    outputs=[gr.Textbox(label="Answer"), gr.JSON(label="Agent decisions")],
    title="ML RAG Assistant",
    description="Agentic retrieval over Karpathy, Ruder, and Colah: retrieve, retry, then answer."
)

if __name__ == "__main__":
    demo.launch()