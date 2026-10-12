import importlib.util
import builtins
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch


class GradioAgentTests(unittest.TestCase):
    def test_interface_calls_agent_and_exposes_trace(self):
        agent = Mock(return_value=("answer", [{"id": 2}], {
            "route": "grounded", "trace": [{"attempt": 0}]
        }))
        gradio = Mock()
        with patch.dict(sys.modules, {
            "gradio": gradio,
            "src.generation": Mock(),
            "src.rag_core": Mock(query_rag=Mock(return_value=("plain answer", []))),
        }), patch("src.agent.agentic_query", agent):
            spec = importlib.util.spec_from_file_location(
                "tested_hf_app", Path(__file__).parents[1] / "hf_app.py"
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            result = module.ask("How does SGD work?")
        agent.assert_called_once_with("How does SGD work?")
        self.assertEqual(result, ("answer", {
            "route": "grounded", "trace": [{"attempt": 0}], "retrieved_ids": [2]
        }))


    def test_startup_defers_retrieval_until_gpu_handler(self):
        for device in ("cpu", "cuda"):
            with self.subTest(device=device):
                imports = []
                inside_gpu = []
                original_import = builtins.__import__

                def tracked_import(name, *args, **kwargs):
                    imports.append(name)
                    return original_import(name, *args, **kwargs)

                def gpu(function):
                    def wrapped(question):
                        inside_gpu.append(True)
                        try:
                            return function(question)
                        finally:
                            inside_gpu.pop()
                    return wrapped

                def agent(question):
                    self.assertEqual(bool(inside_gpu), device == "cuda")
                    return "answer", [], {"route": "ungrounded", "trace": []}

                spaces = Mock(GPU=Mock(side_effect=gpu))
                with patch.dict(os.environ, {"RAG_DEVICE": device}), patch.dict(sys.modules, {
                    "gradio": Mock(), "spaces": spaces,
                    "src.generation": Mock(), "src.rag_core": Mock(),
                }), patch("builtins.__import__", side_effect=tracked_import), patch(
                    "src.agent.agentic_query", side_effect=agent
                ):
                    spec = importlib.util.spec_from_file_location(
                        "tested_startup", Path(__file__).parents[1] / "hf_app.py"
                    )
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    self.assertNotIn("src.rag_core", imports)
                    self.assertIn("src.generation", imports)
                    self.assertEqual(module.ask("question")[0], "answer")
                self.assertEqual(spaces.GPU.call_count, int(device == "cuda"))


if __name__ == "__main__":
    unittest.main()
