import importlib.util
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


if __name__ == "__main__":
    unittest.main()
