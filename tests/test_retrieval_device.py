import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, patch


class RetrievalDeviceTests(unittest.TestCase):
    def test_embeddings_stay_on_cpu_outside_gpu_handlers(self):
        sentence_transformers = MagicMock()
        with patch.dict(sys.modules, {
            "sentence_transformers": sentence_transformers,
            "numpy": MagicMock(),
            "faiss": MagicMock(),
        }):
            spec = importlib.util.spec_from_file_location(
                "tested_retrieval", Path(__file__).parents[1] / "src/retrieval.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.Retriever([{"text": "context", "source": "test"}])
        sentence_transformers.SentenceTransformer.assert_called_once_with(
            "all-MiniLM-L6-v2", device="cpu"
        )


if __name__ == "__main__":
    unittest.main()
