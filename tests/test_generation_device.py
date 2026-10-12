import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import MagicMock, Mock, patch


class GenerationDeviceTests(unittest.TestCase):
    def test_generation_and_rewrite_use_selected_device(self):
        for device in ("cpu", "cuda"):
            with self.subTest(device=device):
                transformers = MagicMock()
                torch = MagicMock()
                spaces = Mock()
                spaces.GPU.side_effect = lambda fn: fn
                with patch.dict(os.environ, {"RAG_DEVICE": device}), patch.dict(
                    sys.modules, {"torch": torch, "transformers": transformers, "spaces": spaces}
                ):
                    spec = importlib.util.spec_from_file_location(
                        "tested_generation", Path(__file__).parents[1] / "src/generation.py"
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    module.generate_answer("question", "context")
                    module.rewrite_query("question")
                module.model.to.assert_called_once_with(device)
                self.assertEqual(module.tokenizer.return_value.to.call_count, 2)
                module.tokenizer.return_value.to.assert_called_with(device)
                self.assertEqual(spaces.GPU.call_count, 2 if device == "cuda" else 0)


if __name__ == "__main__":
    unittest.main()
