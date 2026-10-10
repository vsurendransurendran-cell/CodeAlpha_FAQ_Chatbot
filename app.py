"""Flask application: web page + JSON API for the FAQ chatbot.

Run locally:
    python app.py        (Windows PowerShell)
Then open http://127.0.0.1:5000

All matching logic lives in chatbot.py. This file only handles HTTP.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from flask import Flask, jsonify, render_template, request

from chatbot import (
    DEFAULT_THRESHOLD,
    FAQChatbot,
    FAQDataError,
    InvalidQuestionError,
    UnknownCategoryError,
    UsageStats,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FAQ_PATH = BASE_DIR / "data" / "faqs.json"
MAX_QUESTION_LENGTH = 500  # characters; longer messages are rejected

logger = logging.getLogger(__name__)


def create_app(faq_path: Optional[str | Path] = None, threshold: Optional[float] = None) -> Flask:
    """Application factory (also makes the app easy to test)."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # reject huge request bodies

    faq_path = Path(faq_path or os.environ.get("FAQ_FILE", DEFAULT_FAQ_PATH))
    if threshold is None:
        threshold = float(os.environ.get("FAQ_THRESHOLD", DEFAULT_THRESHOLD))

    bot: Optional[FAQChatbot] = None
    load_error: Optional[str] = None
    try:
        bot = FAQChatbot(faq_path, threshold=threshold)
        if bot.skipped:
            logger.warning("%d malformed FAQ entries were skipped", len(bot.skipped))
    except (FAQDataError, ValueError) as exc:
        load_error = str(exc)
        logger.error("Chatbot could not start: %s", load_error)

    stats = UsageStats()

    def error(message: str, status: int):
        return jsonify({"error": message}), status

    def unavailable():
        return error("The chatbot is temporarily unavailable. Please try again later.", 503)

    def json_body() -> Optional[dict[str, Any]]:
        data = request.get_json(silent=True)
        return data if isinstance(data, dict) else None

    @app.get("/")
    def index():
        return render_template("index.html", max_length=MAX_QUESTION_LENGTH)

    @app.get("/api/health")
    def health():
        if bot is None:
            return jsonify({"status": "unavailable", "faq_count": 0}), 503
        return jsonify(
            {
                "status": "ok",
                "faq_count": len(bot.faqs),
                "categories": len(bot.categories),
                "skipped_entries": len(bot.skipped),
            }
        )

    @app.get("/api/config")
    def config():
        if bot is None:
            return unavailable()
        return jsonify(
            {
                "categories": bot.categories,
                "suggestions": bot.suggested_questions(6),
                "max_length": MAX_QUESTION_LENGTH,
            }
        )

    @app.post("/api/ask")
    def ask():
        if bot is None:
            return unavailable()
        data = json_body()
        if data is None:
            return error("Request must be JSON like {\"question\": \"...\"}.", 400)

        question = data.get("question")
        if not isinstance(question, str):
            return error("'question' must be text.", 400)
        question = question.strip()
        if not question:
            return error("Please type a question.", 400)
        if len(question) > MAX_QUESTION_LENGTH:
            return error(f"Please keep your question under {MAX_QUESTION_LENGTH} characters.", 400)

        category = data.get("category") or None
        if category is not None and not isinstance(category, str):
            return error("'category' must be text.", 400)

        try:
            result = bot.ask(question, category=category)
        except InvalidQuestionError as exc:
            return error(str(exc), 400)
        except UnknownCategoryError as exc:
            return error(str(exc), 400)
        except Exception:  # never leak internal details to the user
            logger.exception("Unexpected error while answering a question")
            return error("Something went wrong while answering. Please try again.", 500)

        stats.record_question(result.matched, result.faq_id)
        return jsonify(result.to_dict())

    @app.post("/api/feedback")
    def feedback():
        if bot is None:
            return unavailable()
        data = json_body()
        if data is None:
            return error("Request must be JSON.", 400)
        faq_id, helpful = data.get("faq_id"), data.get("helpful")
        if not isinstance(faq_id, str) or bot.get_faq(faq_id) is None:
            return error("Unknown FAQ id.", 400)
        if not isinstance(helpful, bool):
            return error("'helpful' must be true or false.", 400)
        stats.record_feedback(faq_id, helpful)
        return jsonify({"status": "recorded"})

    @app.get("/api/stats")
    def get_stats():
        if bot is None:
            return unavailable()
        return jsonify(stats.snapshot(bot))

    @app.errorhandler(404)
    def not_found(_):
        if request.path.startswith("/api/"):
            return error("Not found.", 404)
        return "Page not found.", 404

    @app.errorhandler(405)
    def method_not_allowed(_):
        return error("Method not allowed.", 405)

    @app.errorhandler(413)
    def too_large(_):
        return error("Request is too large.", 413)

    app.extensions["faq_bot"] = bot
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # debug=False by default; set FLASK_DEBUG=1 for auto-reload while developing.
    create_app().run(host="127.0.0.1", port=5000, debug=os.environ.get("FLASK_DEBUG") == "1")
