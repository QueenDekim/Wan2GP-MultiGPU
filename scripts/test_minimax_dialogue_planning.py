#!/usr/bin/env python3
"""Dependency-free regression tests for MiniMax H3 multilingual audio duration.

Runs in GitHub Actions without downloading torch, torchaudio, or checkpoints.
Only dialogue parsing and duration planning are invoked; torch imports are
stubbed because the other audio/Whisper functions are not used by these tests.
"""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_dialogue():
    filename = Path(__file__).resolve().parents[1] / "models/minimax_h3/dialogue.py"
    module_name = "_h3_dialogue_planning_regression"
    torch = types.ModuleType("torch")
    torchaudio = types.ModuleType("torchaudio")
    torchaudio.__path__ = []
    functional = types.ModuleType("torchaudio.functional")
    torchaudio.functional = functional

    spec = importlib.util.spec_from_file_location(module_name, filename)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        "torch": torch,
        "torchaudio": torchaudio,
        "torchaudio.functional": functional,
        module_name: module,
    }):
        spec.loader.exec_module(module)
    return module


DIALOGUE = load_dialogue()


class DialogueDurationTest(unittest.TestCase):
    def test_cyrillic_words_survive_normalization(self):
        phrase = "Привет, мир! На снежном холме стоит женщина."
        self.assertEqual(
            DIALOGUE._normalize_words(phrase),
            ["привет", "мир", "на", "снежном", "холме", "стоит", "женщина"],
        )

    def test_multilingual_aligner_uses_same_unicode_words(self):
        russian = "На горизонте виден лес и огромная река."
        words = DIALOGUE._normalize_words(russian)
        alignment = DIALOGUE._align_words(words, words)
        self.assertEqual(len(alignment), len(words))
        self.assertTrue(all(label == "match" for label, _ in alignment))

    def test_russian_monologue_is_not_four_seconds(self):
        phrase = (
            "Камера плавно поворачивается вокруг женщины и отдаляется, "
            "открывая перед ней бескрайнее заснеженное поле. "
            "На горизонте виднеется лес и река, а вдалеке стоит "
            "небольшая белая церковь с золотым куполом."
        )
        turns = DIALOGUE.plan_dialogue(
            "Speaker 1:\n[Russian, thoughtful and close to the microphone] " + phrase,
            seed=1234,
            duration_seconds=120,
        )
        self.assertEqual(len(turns), 1)
        self.assertGreater(turns[0].duration_s, 10)
        self.assertLess(turns[0].duration_s, 120)
        self.assertEqual(turns[0].language_code, "ru")

    def test_long_russian_monologue_is_split_without_losing_text(self):
        phrase = " ".join("заснеженный простор и величественная река" for _ in range(60))
        turns = DIALOGUE.plan_dialogue(
            "Speaker 1:\n[Russian, calm] " + phrase,
            seed=77,
            duration_seconds=120,
        )
        self.assertGreater(len(turns), 1)
        self.assertEqual(" ".join(turn.text for turn in turns), phrase)
        self.assertTrue(all(turn.speaker == 1 for turn in turns))
        self.assertTrue(all(turn.language_code == "ru" for turn in turns))
        self.assertTrue(all(4 <= turn.duration_s <= 45 for turn in turns))
        total = (sum(turn.duration_s for turn in turns)
                 + (len(turns) - 1) * DIALOGUE.H3_DIALOGUE_PAUSE_SECONDS)
        self.assertLessEqual(total, 120.001)
        self.assertGreater(total, 90)
        self.assertEqual([turn.seed for turn in turns],
                         [77 + i * 1000 for i in range(len(turns))])

    def test_total_duration_is_a_cap_not_target(self):
        prompt = "Speaker 1: Привет, это короткая реплика."
        turns = DIALOGUE.plan_dialogue(prompt, seed=1, duration_seconds=120)
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0].duration_s, 4)

    def test_english_behavior(self):
        self.assertEqual(DIALOGUE._normalize_words("It's a short scene."),
                         ["it's", "a", "short", "scene"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
