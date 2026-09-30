#!/usr/bin/env python3

import json
import signal
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import evaluate_api_bank as evaluator


class ApiBankEvaluatorTest(unittest.TestCase):
    def test_data_parallel_cleanup_signals_process_groups(self):
        process = mock.Mock()
        process.pid = 12345
        process.is_alive.return_value = False

        with mock.patch.object(evaluator.os, "name", "posix"), mock.patch.object(
            evaluator.os, "killpg"
        ) as killpg:
            evaluator.terminate_data_parallel_processes([process], timeout=0)

        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(12345, signal.SIGTERM),
                mock.call(12345, signal.SIGKILL),
            ],
        )
        process.terminate.assert_not_called()
        process.kill.assert_not_called()

    def test_parallel_config_validation(self):
        self.assertEqual(evaluator.validate_parallel_config(4, 2, "0,1,2,3,4,5,6,7"), 8)
        with self.assertRaisesRegex(ValueError, "requires 8 GPUs"):
            evaluator.validate_parallel_config(4, 2, "0,1,2,3")
        with self.assertRaisesRegex(ValueError, "data_parallel_size"):
            evaluator.validate_parallel_config(0, 1, "0")

        shards = evaluator.partition_indexed_items(["a", "b", "c", "d", "e"], 2)
        self.assertEqual(shards, [[(0, "a"), (2, "c"), (4, "e")], [(1, "b"), (3, "d")]])
        restored = [item for _, item in sorted(shards[0] + shards[1])]
        self.assertEqual(restored, ["a", "b", "c", "d", "e"])

    def test_parse_official_api_call(self):
        parsed = evaluator.parse_official_api_call(
            "API-Request: [Example(query='a(b)', count=2, flags=[True, None])]"
        )
        self.assertEqual(parsed["name"], "Example")
        self.assertEqual(
            parsed["parameters"],
            {"query": "a(b)", "count": 2, "flags": [True, None]},
        )

        placeholder = evaluator.parse_official_api_call(
            "API-Request: [Example(key=..., nested=[..., 1])]"
        )
        self.assertEqual(
            placeholder["parameters"], {"key": "...", "nested": ["...", 1]}
        )
        json.dumps(placeholder)

    def test_parse_upstream_unescaped_list_value(self):
        parsed = evaluator.parse_official_api_call(
            "API-Request: [AddMeeting(topic='Team', attendees='['Mary', 'Peter']')]"
        )
        self.assertEqual(
            parsed,
            {
                "name": "AddMeeting",
                "parameters": {"topic": "Team", "attendees": ["Mary", "Peter"]},
            },
        )

    def test_parse_concatenated_rlla_calls(self):
        text = (
            "<think>test</think><tool_call>\n"
            '{"name":"A","parameters":{"x":1}}\n'
            '{"name":"B","parameters":{}}\n'
            "</tool_call>"
        )
        calls, valid = evaluator.parse_rlla_tool_calls(text)
        self.assertTrue(valid)
        self.assertEqual([call["name"] for call in calls], ["A", "B"])

    def test_qwen_prefilled_think_prefix_is_valid_rlla(self):
        prompt = "<|im_start|>assistant\n<think>\n"
        prediction = (
            "Reasoning supplied after the prompt prefix.\n</think>\n"
            '<tool_call>{"name":"Lookup","parameters":{"name":"John"}}'
            "</tool_call>"
        )
        reference = "API-Request: [Lookup(name='John')]"

        self.assertTrue(evaluator.prompt_prefills_rlla_think(prompt))
        self.assertFalse(evaluator.rlla_format_valid(prediction))
        score = evaluator.score_api_prediction(
            prediction,
            reference,
            prompt_template="rlla",
            prompt_prefills_think=True,
        )
        self.assertTrue(score["rlla_format_valid"])
        self.assertTrue(score["call_exact_match"])

        missing_close = prediction.replace("</think>", "")
        self.assertFalse(
            evaluator.rlla_format_valid(
                missing_close, prompt_prefills_think=True
            )
        )

    def test_prefilled_think_response_fallback_drops_reasoning(self):
        reference = "The operation succeeded."
        score = evaluator.score_response_prediction(
            "Private reasoning.\n</think>\nThe operation succeeded.",
            reference,
            prompt_template="rlla",
            prompt_prefills_think=True,
        )
        self.assertEqual(score["extracted_response"], reference)
        self.assertEqual(score["rouge_l_f1"], 1.0)

    def test_gold_api_score(self):
        reference = "API-Request: [Lookup(name='John')]"
        prediction = evaluator.gold_prediction("api", reference)
        score = evaluator.score_api_prediction(prediction, reference)
        self.assertTrue(score["call_exact_match"])
        self.assertTrue(score["rlla_format_valid"])

    def test_official_api_output_score(self):
        reference = "API-Request: [Lookup(name='John')]"
        prediction = "API-Request: [Lookup(name='John')]"
        score = evaluator.score_api_prediction(
            prediction, reference, prompt_template="official"
        )
        self.assertTrue(score["call_exact_match"])
        self.assertTrue(score["official_format_valid"])
        self.assertTrue(score["api_call_valid"])

    def test_response_score(self):
        reference = "The operation succeeded."
        prediction = evaluator.gold_prediction("response", reference)
        score = evaluator.score_response_prediction(prediction, reference)
        self.assertEqual(score["rouge_l_f1"], 1.0)
        self.assertTrue(score["exact_match"])

    def test_official_response_score(self):
        reference = "The operation succeeded."
        score = evaluator.score_response_prediction(
            "AI: The operation succeeded.",
            reference,
            prompt_template="official",
        )
        self.assertEqual(score["extracted_response"], reference)
        self.assertEqual(score["rouge_l_f1"], 1.0)
        self.assertTrue(score["official_format_valid"])

    def test_rlla_prompt_adapter(self):
        tool = {
            "name": "Lookup",
            "description": "Look up a user.",
            "input_parameters": {"name": {"type": "str"}},
            "output_parameters": {},
        }
        entry = {
            "instruction": "API descriptions:\n" + json.dumps(tool),
            "input": (
                "User: Find John.\n"
                "AI: I will look him up.\n"
                "API-Request: [Lookup(name='John')]->{'id': 1}\n"
                "User: What is his id?\n"
                "Generate AI Response:\n"
            ),
        }
        messages = evaluator.build_rlla_messages(entry)
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("1. Name: Lookup", messages[0]["content"])
        self.assertIn("**Dialogue Records History**", messages[1]["content"])
        self.assertIn("<tool_call>", messages[1]["content"])
        self.assertIn("<obs>", messages[1]["content"])
        self.assertIn("<user> What is his id? </user>", messages[1]["content"])

    def test_official_prompt_adapter(self):
        entry = {
            "instruction": "Official instruction.\nThe current year is 2023.\n",
            "input": "\nUser: Find John.\nGenerate API Request:\n",
        }
        messages = evaluator.build_prompt_messages(entry, "official")
        self.assertEqual(
            messages,
            [
                {
                    "role": "user",
                    "content": (
                        "Official instruction.\nThe current year is 2023.\n"
                        "User: Find John.\nGenerate API Request:\n"
                    ),
                }
            ],
        )

        class ChatTemplateMustNotRun:
            def apply_chat_template(self, *args, **kwargs):
                raise AssertionError("official prompt was wrapped in a chat template")

        rendered = evaluator.render_model_prompt(
            messages, "official", ChatTemplateMustNotRun()
        )
        self.assertEqual(rendered, messages[0]["content"])

    def test_tokenizer_config_is_repaired_without_mutating_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = {
                "tokenizer_class": "ExampleTokenizer",
                "extra_special_tokens": ["<special>"],
            }
            (root / "tokenizer_config.json").write_text(json.dumps(original))
            source, temporary = evaluator.prepare_tokenizer_source(str(root))
            self.assertIsNotNone(temporary)
            repaired = json.loads((Path(source) / "tokenizer_config.json").read_text())
            self.assertNotIn("extra_special_tokens", repaired)
            self.assertEqual(repaired["additional_special_tokens"], ["<special>"])
            self.assertEqual(
                json.loads((root / "tokenizer_config.json").read_text()), original
            )
            temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
