"""FAQ chatbot engine: loading, text processing, matching and answer selection.

This module contains NO Flask code. It can be used (and tested) on its own:

    bot = FAQChatbot("data/faqs.json")
    print(bot.ask("what is python").to_dict())

How matching works (details in README.md):
1. The question is normalised: lowercased, punctuation and filler words removed.
2. It is turned into two TF-IDF vectors: one over words, one over character
   n-grams (the character vector makes small spelling mistakes tolerable).
3. Cosine similarity compares it with every stored FAQ phrase.
4. The best score is the "confidence". If it is below the threshold, the bot
   returns a fallback message instead of guessing.

This is retrieval, not text generation: answers are always taken from the JSON file.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Settings (the threshold can also be overridden when creating the bot)
# --------------------------------------------------------------------------
DEFAULT_THRESHOLD = 0.40  # minimum confidence (0-1) needed to give an answer
DEFAULT_RELATED_THRESHOLD = 0.15  # minimum score for a "related question"
DEFAULT_MAX_RELATED = 3

# How the two similarity signals are mixed (must add up to 1.0).
WORD_WEIGHT = 0.5
CHAR_WEIGHT = 0.5

FALLBACK_MESSAGE = (
    "Sorry, I couldn't find a confident answer to that in my FAQ. "
    "Try rephrasing with different keywords, or pick one of the suggestions below."
)

# Small hand-made stop-word list. We do not use scikit-learn's built-in list
# because it removes useful words such as "system" and "computer".
STOP_WORDS = frozenset(
    """
    a an the is are was were be been am do does did of to in on for with and or
    what whats how why who when where which can could should would will shall
    i me my you your we our it its this that these those there about please
    tell explain define mean means between
    """.split()
)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------
class FAQDataError(Exception):
    """The knowledge base is missing, unreadable, or contains no usable FAQs."""


class InvalidQuestionError(ValueError):
    """The user's question is empty or contains no usable words."""


class UnknownCategoryError(ValueError):
    """A category filter was requested that does not exist."""


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FAQEntry:
    id: str
    category: str
    question: str
    answer: str
    alternatives: tuple[str, ...] = ()


@dataclass
class ChatResult:
    """What the engine returns for one question."""

    answer: str
    matched: bool
    confidence: float
    faq_id: Optional[str] = None
    category: Optional[str] = None
    matched_question: Optional[str] = None
    related: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "matched": self.matched,
            "confidence": round(self.confidence, 3),
            "faq_id": self.faq_id,
            "category": self.category,
            "matched_question": self.matched_question,
            "related": self.related,
        }


# --------------------------------------------------------------------------
# Text processing
# --------------------------------------------------------------------------
def preprocess(text: str) -> str:
    """Normalise text for matching.

    Lowercase, drop apostrophes, replace punctuation with spaces (keeping + and #
    so "C++" and "C#" survive) and remove stop words. If *every* word is a stop
    word (e.g. "who are you") the original words are kept so the text is not empty.
    """
    text = text.lower()
    text = re.sub(r"['’`]", "", text)  # what's -> whats
    text = text.replace("_", " ")
    text = re.sub(r"[^\w\s+#]", " ", text)
    tokens = text.split()
    content = [t for t in tokens if t not in STOP_WORDS]
    return " ".join(content or tokens)


def _validate_entry(raw: Any, seen_ids: set[str]) -> FAQEntry:
    """Turn one raw JSON object into an FAQEntry or raise ValueError explaining why not."""
    if not isinstance(raw, dict):
        raise ValueError("entry is not a JSON object")

    def required_text(key: str) -> str:
        value = raw.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"missing or empty '{key}'")
        return value.strip()

    entry_id = required_text("id")
    category = required_text("category")
    question = required_text("question")
    answer = required_text("answer")

    if entry_id in seen_ids:
        raise ValueError(f"duplicate id '{entry_id}'")
    if not preprocess(question):
        raise ValueError("question contains no usable words")

    alternatives_raw = raw.get("alternative_questions", [])
    if alternatives_raw is None:
        alternatives_raw = []
    if not isinstance(alternatives_raw, list) or not all(
        isinstance(a, str) and a.strip() for a in alternatives_raw
    ):
        raise ValueError("'alternative_questions' must be a list of non-empty strings")
    alternatives = tuple(a.strip() for a in alternatives_raw if preprocess(a))

    return FAQEntry(entry_id, category, question, answer, alternatives)


def load_faqs(path: str | Path) -> tuple[list[FAQEntry], list[str]]:
    """Read and validate the knowledge base.

    Returns (faqs, skipped) where *skipped* describes malformed entries that were
    ignored. Raises FAQDataError if the file is missing/unreadable/invalid or if
    no valid entries remain.
    """
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise FAQDataError(f"FAQ file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FAQDataError(f"FAQ file is not valid JSON ({exc})") from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise FAQDataError(f"FAQ file could not be read: {exc}") from exc

    raw_entries = data.get("faqs") if isinstance(data, dict) else data
    if not isinstance(raw_entries, list):
        raise FAQDataError("FAQ file must contain a list of entries under the key 'faqs'")

    faqs: list[FAQEntry] = []
    skipped: list[str] = []
    seen_ids: set[str] = set()
    for position, raw in enumerate(raw_entries, start=1):
        try:
            entry = _validate_entry(raw, seen_ids)
        except ValueError as exc:
            message = f"entry #{position} skipped: {exc}"
            logger.warning(message)
            skipped.append(message)
            continue
        seen_ids.add(entry.id)
        faqs.append(entry)

    if not faqs:
        raise FAQDataError("FAQ file contains no valid entries")
    return faqs, skipped


# --------------------------------------------------------------------------
# The chatbot
# --------------------------------------------------------------------------
class FAQChatbot:
    """Retrieval-based FAQ matcher using TF-IDF and cosine similarity."""

    def __init__(
        self,
        faq_path: str | Path,
        threshold: float = DEFAULT_THRESHOLD,
        related_threshold: float = DEFAULT_RELATED_THRESHOLD,
        max_related: int = DEFAULT_MAX_RELATED,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        self.threshold = threshold
        self.related_threshold = related_threshold
        self.max_related = max_related

        self.faqs, self.skipped = load_faqs(faq_path)
        self._by_id = {faq.id: faq for faq in self.faqs}
        self._build_index()

    # ---- index construction ------------------------------------------------
    def _build_index(self) -> None:
        """Vectorise every FAQ phrase (the question plus its alternatives)."""
        phrases: list[str] = []
        owners: list[int] = []  # phrase index -> FAQ index
        for faq_index, faq in enumerate(self.faqs):
            for text in (faq.question, *faq.alternatives):
                phrases.append(preprocess(text))
                owners.append(faq_index)
        self._owners = np.array(owners)
        # Vocabulary used to repair spelling mistakes (see _correct_spelling).
        self._vocabulary = sorted({tok for phrase in phrases for tok in phrase.split()})

        token_pattern = r"(?u)[\w+#]+"  # keeps one-letter words and C++ / C#
        self._word_vectorizer = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), token_pattern=token_pattern, sublinear_tf=True
        )
        self._char_vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 4), sublinear_tf=True
        )
        self._word_matrix = self._word_vectorizer.fit_transform(phrases)
        self._char_matrix = self._char_vectorizer.fit_transform(phrases)

    # ---- spelling ---------------------------------------------------------------
    def _correct_spelling(self, processed_question: str) -> str:
        """Replace unknown words with the closest known FAQ word (e.g. pyhton -> python).

        Only words of 4+ letters that are not already in the vocabulary are touched,
        and only when a very similar word exists (difflib ratio >= 0.8), so unrelated
        words are left alone and still end up in the fallback.
        """
        known = set(self._vocabulary)
        fixed = []
        for token in processed_question.split():
            if token not in known and len(token) >= 4 and token.isalpha():
                close = difflib.get_close_matches(token, self._vocabulary, n=1, cutoff=0.8)
                token = close[0] if close else token
            fixed.append(token)
        return " ".join(fixed)

    # ---- scoring ------------------------------------------------------------
    def _score_faqs(self, processed_question: str) -> np.ndarray:
        """Return one similarity score (0-1) per FAQ: the best of its phrases."""
        word_scores = cosine_similarity(
            self._word_vectorizer.transform([processed_question]), self._word_matrix
        ).ravel()
        char_scores = cosine_similarity(
            self._char_vectorizer.transform([processed_question]), self._char_matrix
        ).ravel()
        phrase_scores = WORD_WEIGHT * word_scores + CHAR_WEIGHT * char_scores

        faq_scores = np.zeros(len(self.faqs))
        np.maximum.at(faq_scores, self._owners, phrase_scores)
        return faq_scores

    # ---- public helpers -------------------------------------------------------
    @property
    def categories(self) -> list[str]:
        """Category names in the order they first appear in the file."""
        return list(dict.fromkeys(faq.category for faq in self.faqs))

    def get_faq(self, faq_id: str) -> Optional[FAQEntry]:
        return self._by_id.get(faq_id)

    def suggested_questions(self, count: int = 6) -> list[dict[str, str]]:
        """Pick example questions, one per category first, so the list is varied."""
        by_category: dict[str, list[FAQEntry]] = {}
        for faq in self.faqs:
            by_category.setdefault(faq.category, []).append(faq)
        picked: list[FAQEntry] = []
        round_number = 0
        while len(picked) < count and any(len(v) > round_number for v in by_category.values()):
            for entries in by_category.values():
                if len(entries) > round_number and len(picked) < count:
                    picked.append(entries[round_number])
            round_number += 1
        return [{"id": f.id, "question": f.question, "category": f.category} for f in picked]

    # ---- main entry point -----------------------------------------------------
    def ask(self, question: str, category: Optional[str] = None) -> ChatResult:
        """Answer *question*, optionally restricted to one *category*."""
        if not isinstance(question, str) or not question.strip():
            raise InvalidQuestionError("Please type a question.")
        processed = preprocess(question)
        if not processed:
            raise InvalidQuestionError("Please include some words in your question.")

        if category:
            if category not in self.categories:
                raise UnknownCategoryError(f"Unknown category: {category}")
            candidates = [i for i, faq in enumerate(self.faqs) if faq.category == category]
        else:
            candidates = list(range(len(self.faqs)))

        scores = self._score_faqs(self._correct_spelling(processed))
        ranked = sorted(candidates, key=lambda i: scores[i], reverse=True)
        best_index = ranked[0]
        confidence = float(scores[best_index])
        matched = confidence >= self.threshold

        if matched:
            best = self.faqs[best_index]
            related = self._related(ranked, scores, exclude=best_index, fallback_category=best.category)
            return ChatResult(
                answer=best.answer,
                matched=True,
                confidence=confidence,
                faq_id=best.id,
                category=best.category,
                matched_question=best.question,
                related=related,
            )

        # No confident match: do NOT guess. Offer the closest questions instead.
        related = self._related(ranked, scores, exclude=None, fallback_category=None)
        if not related:
            related = [
                {"id": s["id"], "question": s["question"]}
                for s in self.suggested_questions(self.max_related)
            ]
        return ChatResult(
            answer=FALLBACK_MESSAGE,
            matched=False,
            confidence=confidence,
            related=related,
        )

    def _related(
        self,
        ranked: list[int],
        scores: np.ndarray,
        exclude: Optional[int],
        fallback_category: Optional[str],
    ) -> list[dict[str, str]]:
        """Closest other FAQs above the related threshold (else same-category ones)."""
        close = [
            i for i in ranked if i != exclude and scores[i] >= self.related_threshold
        ][: self.max_related]
        if not close and fallback_category:
            close = [
                i
                for i, faq in enumerate(self.faqs)
                if faq.category == fallback_category and i != exclude
            ][: self.max_related]
        return [{"id": self.faqs[i].id, "question": self.faqs[i].question} for i in close]


# --------------------------------------------------------------------------
# Simple, privacy-friendly usage statistics (in memory only)
# --------------------------------------------------------------------------
class UsageStats:
    """Thread-safe counters. Stores NO message text, only counts per FAQ id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._answered = 0
        self._unanswered = 0
        self._matches: Counter[str] = Counter()
        self._helpful: Counter[str] = Counter()
        self._not_helpful: Counter[str] = Counter()

    def record_question(self, matched: bool, faq_id: Optional[str]) -> None:
        with self._lock:
            if matched and faq_id:
                self._answered += 1
                self._matches[faq_id] += 1
            else:
                self._unanswered += 1

    def record_feedback(self, faq_id: str, helpful: bool) -> None:
        with self._lock:
            (self._helpful if helpful else self._not_helpful)[faq_id] += 1

    def snapshot(self, bot: FAQChatbot, top_n: int = 5) -> dict[str, Any]:
        with self._lock:
            total = self._answered + self._unanswered
            top = []
            for faq_id, count in self._matches.most_common(top_n):
                faq = bot.get_faq(faq_id)
                if faq:
                    top.append({"id": faq_id, "question": faq.question, "count": count})
            return {
                "total_questions": total,
                "answered": self._answered,
                "unanswered": self._unanswered,
                "answer_rate": round(self._answered / total, 3) if total else None,
                "top_questions": top,
                "feedback_helpful": sum(self._helpful.values()),
                "feedback_not_helpful": sum(self._not_helpful.values()),
            }
