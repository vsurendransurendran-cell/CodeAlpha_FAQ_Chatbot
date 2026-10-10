"""Automated tests (standard library unittest, no extra packages needed).

Run from the project folder:
    python -m unittest discover -s tests -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import MAX_QUESTION_LENGTH, create_app  # noqa: E402
from chatbot import (  # noqa: E402
    FALLBACK_MESSAGE,
    FAQChatbot,
    FAQDataError,
    InvalidQuestionError,
    UnknownCategoryError,
    load_faqs,
    preprocess,
)

FAQ_FILE = ROOT / "data" / "faqs.json"


def write_json(directory: str, name: str, payload) -> Path:
    path = Path(directory) / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class PreprocessTests(unittest.TestCase):
    def test_lowercases_and_strips_punctuation_and_stop_words(self):
        self.assertEqual(preprocess("What IS Python?!"), "python")

    def test_keeps_special_language_names(self):
        self.assertIn("c++", preprocess("Is C++ hard?"))

    def test_all_stop_words_not_empty(self):
        self.assertTrue(preprocess("who are you"))


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = FAQChatbot(FAQ_FILE)

    def test_exact_question_returns_its_own_answer(self):
        for faq in self.bot.faqs:
            with self.subTest(faq=faq.id):
                result = self.bot.ask(faq.question)
                self.assertTrue(result.matched)
                self.assertEqual(result.faq_id, faq.id)

    def test_alternative_phrasings_match_their_faq(self):
        for faq in self.bot.faqs:
            for alt in faq.alternatives:
                with self.subTest(faq=faq.id, alt=alt):
                    result = self.bot.ask(alt)
                    self.assertTrue(result.matched)
                    self.assertEqual(result.faq_id, faq.id)

    def test_rephrased_question(self):
        result = self.bot.ask("can you tell me what python is")
        self.assertTrue(result.matched)
        self.assertIn("python", result.matched_question.lower())

    def test_spelling_variation(self):
        for typo, expected in [
            ("what is machin lerning", "machine learning"),
            ("what is pyhton", "python"),
            ("what is artifical inteligence", "artificial intelligence"),
        ]:
            with self.subTest(typo=typo):
                result = self.bot.ask(typo)
                self.assertTrue(result.matched)
                self.assertIn(expected, result.matched_question.lower())

    def test_unrelated_question_gets_fallback(self):
        for q in ["what is the price of a pizza in rome", "zxqv blorp wibble"]:
            with self.subTest(q=q):
                result = self.bot.ask(q)
                self.assertFalse(result.matched)
                self.assertEqual(result.answer, FALLBACK_MESSAGE)
                self.assertTrue(result.related)  # still offers suggestions

    def test_repeated_question_gives_same_answer(self):
        first = self.bot.ask("What is Python?")
        second = self.bot.ask("What is Python?")
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_empty_questions_rejected(self):
        for q in ["", "   ", "?!?", None, 42]:
            with self.subTest(q=q):
                with self.assertRaises(InvalidQuestionError):
                    self.bot.ask(q)

    def test_category_filter(self):
        category = self.bot.categories[0]
        result = self.bot.ask("what is python", category=category)
        if result.matched:
            self.assertEqual(result.category, category)
        with self.assertRaises(UnknownCategoryError):
            self.bot.ask("what is python", category="Nope")

    def test_related_excludes_best_match(self):
        result = self.bot.ask("What is Python?")
        self.assertNotIn(result.faq_id, [r["id"] for r in result.related])

    def test_threshold_changes_behaviour(self):
        unrelated = "zxqv blorp wibble"
        self.assertFalse(FAQChatbot(FAQ_FILE, threshold=0.4).ask(unrelated).matched)
        self.assertTrue(FAQChatbot(FAQ_FILE, threshold=0.0).ask(unrelated).matched)
        with self.assertRaises(ValueError):
            FAQChatbot(FAQ_FILE, threshold=1.5)

    def test_suggestions_have_expected_shape(self):
        suggestions = self.bot.suggested_questions(6)
        self.assertEqual(len(suggestions), 6)
        self.assertTrue(all({"id", "question", "category"} <= set(s) for s in suggestions))


class KnowledgeBaseTests(unittest.TestCase):
    def test_real_file_is_valid_and_has_no_skipped_entries(self):
        faqs, skipped = load_faqs(FAQ_FILE)
        self.assertEqual(skipped, [])
        self.assertGreaterEqual(len({f.category for f in faqs}), 6)

    def test_missing_file(self):
        with self.assertRaises(FAQDataError):
            load_faqs(ROOT / "data" / "does_not_exist.json")

    def test_invalid_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.json"
            path.write_text("{ not json", encoding="utf-8")
            with self.assertRaises(FAQDataError):
                load_faqs(path)

    def test_wrong_structure(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FAQDataError):
                load_faqs(write_json(d, "x.json", {"faqs": "nope"}))

    def test_malformed_entries_skipped_but_valid_kept(self):
        good = {"id": "a", "category": "C", "question": "What is Flask?", "answer": "A web framework."}
        payload = {
            "faqs": [
                good,
                {"id": "b", "category": "C", "question": "", "answer": "x"},  # empty question
                {"id": "a", "category": "C", "question": "Duplicate id", "answer": "x"},
                "not an object",
                {"id": "c", "category": "C", "question": "Bad alts", "answer": "x", "alternative_questions": "oops"},
            ]
        }
        with tempfile.TemporaryDirectory() as d:
            faqs, skipped = load_faqs(write_json(d, "mixed.json", payload))
        self.assertEqual([f.id for f in faqs], ["a"])
        self.assertEqual(len(skipped), 4)

    def test_no_valid_entries(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FAQDataError):
                load_faqs(write_json(d, "e.json", {"faqs": [{"id": "x"}]}))


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = create_app(FAQ_FILE).test_client()

    def post(self, path, payload):
        return self.client.post(path, json=payload)

    def test_index_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"FAQ Assistant", response.data)

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_config(self):
        data = self.client.get("/api/config").get_json()
        self.assertTrue(data["categories"])
        self.assertTrue(data["suggestions"])

    def test_ask_success(self):
        response = self.post("/api/ask", {"question": "What is Python?"})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["matched"])
        self.assertIn("answer", data)

    def test_ask_unmatched_is_still_200_with_fallback(self):
        response = self.post("/api/ask", {"question": "zxqv blorp wibble"})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.get_json()["matched"])

    def test_ask_validation_errors(self):
        cases = [
            {"question": ""},
            {"question": "   "},
            {"question": 5},
            {},
            {"question": "x" * (MAX_QUESTION_LENGTH + 1)},
            {"question": "what is python", "category": "Nope"},
            {"question": "what is python", "category": 3},
        ]
        for payload in cases:
            with self.subTest(payload=str(payload)[:40]):
                response = self.post("/api/ask", payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn("error", response.get_json())

    def test_ask_rejects_non_json(self):
        response = self.client.post("/api/ask", data="question=hi", content_type="text/plain")
        self.assertEqual(response.status_code, 400)
        response = self.client.post("/api/ask", json=["a list"])
        self.assertEqual(response.status_code, 400)

    def test_wrong_method_and_unknown_route(self):
        self.assertEqual(self.client.get("/api/ask").status_code, 405)
        self.assertEqual(self.client.get("/api/nope").status_code, 404)

    def test_feedback_and_stats(self):
        app = create_app(FAQ_FILE)
        client = app.test_client()
        data = client.post("/api/ask", json={"question": "What is Python?"}).get_json()
        client.post("/api/ask", json={"question": "zxqv blorp wibble"})
        ok = client.post("/api/feedback", json={"faq_id": data["faq_id"], "helpful": True})
        self.assertEqual(ok.status_code, 200)
        self.assertEqual(client.post("/api/feedback", json={"faq_id": "nope", "helpful": True}).status_code, 400)
        self.assertEqual(client.post("/api/feedback", json={"faq_id": data["faq_id"], "helpful": "yes"}).status_code, 400)
        stats = client.get("/api/stats").get_json()
        self.assertEqual(stats["total_questions"], 2)
        self.assertEqual(stats["answered"], 1)
        self.assertEqual(stats["unanswered"], 1)
        self.assertEqual(stats["feedback_helpful"], 1)

    def test_app_survives_missing_knowledge_base(self):
        client = create_app(ROOT / "data" / "missing.json").test_client()
        self.assertEqual(client.get("/api/health").status_code, 503)
        response = client.post("/api/ask", json={"question": "hi there"})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("missing.json", response.get_data(as_text=True))  # no internal details
        self.assertEqual(client.get("/").status_code, 200)


if __name__ == "__main__":
    unittest.main()
