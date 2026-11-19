"""
src/explainability.py
=====================
Explainability module for the Context-Aware Mental State Detector.

Provides lightweight, dependency-minimal explanations for every prediction
produced by ``MentalStatePredictor``.  This module is intentionally decoupled
from the inference pipeline — it operates purely on text and the already-
computed prediction dict, so it can be dropped in or removed without touching
any other file.

Usage
-----
    from src.explainability import explain

    result = predictor.predict(current_text="I haven't slept in days…",
                               context_history=[...])
    explanation = explain(text=current_text, prediction=result)
    # → {
    #       "keywords":       ["exhausted", "sleep"],
    #       "sentiment_score": -0.61,
    #       "reason":          "The message contains …"
    #   }
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Sentiment backend — VADER preferred, TextBlob as fallback, plain heuristic
# last resort.  Nothing is hard-required: import errors are caught gracefully.
# ---------------------------------------------------------------------------

def _build_sentiment_analyser():
    """Return a callable (text) → float in [-1, 1]."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
        return lambda text: _vader.polarity_scores(text)["compound"]
    except ImportError:
        pass

    try:
        from textblob import TextBlob
        return lambda text: TextBlob(text).sentiment.polarity
    except ImportError:
        pass

    # Pure-Python fallback: average of simple positive/negative word lists
    _POS = {"good", "great", "happy", "fine", "okay", "well", "love", "joy",
            "wonderful", "calm", "peaceful", "better", "enjoy", "hopeful"}
    _NEG = {"bad", "sad", "terrible", "awful", "hate", "worse", "fear",
            "scared", "angry", "hurt", "pain", "angry", "hopeless", "alone",
            "empty", "worthless", "fail", "failed", "broken", "dark", "lost"}

    def _heuristic(text: str) -> float:
        tokens = set(re.findall(r"[a-z]+", text.lower()))
        pos = len(tokens & _POS)
        neg = len(tokens & _NEG)
        total = pos + neg
        return (pos - neg) / total if total else 0.0

    return _heuristic


_get_sentiment = _build_sentiment_analyser()


# ---------------------------------------------------------------------------
# Keyword lexicons — deliberately concise to stay dependency-free
# ---------------------------------------------------------------------------

_KEYWORD_SETS: dict[str, set[str]] = {
    "stress": {
        "overwhelmed", "pressure", "deadline", "stress", "stressed",
        "exhausted", "burnout", "overloaded", "hectic", "rush",
        "workload", "tension", "frantic", "panic", "anxious", "anxiety",
        "nervous", "worry", "worried", "tense", "tight", "demanding",
        "chaotic", "drained", "tired", "sleepless", "insomnia", "cope",
        "coping", "can't cope", "too much", "no time", "behind",
    },
    "sadness": {
        "sad", "depressed", "depression", "hopeless", "hopelessness",
        "empty", "worthless", "lonely", "alone", "isolated", "numb",
        "miserable", "crying", "cried", "tears", "broken", "dark",
        "darkness", "lost", "meaningless", "pointless", "give up",
        "giving up", "can't go on", "no point", "nothing matters",
        "joyless", "grey", "gray", "hollow", "disconnected",
    },
    "anxiety": {
        "afraid", "fear", "scared", "terrified", "dreading", "dread",
        "panic", "panicking", "worried", "worry", "uneasy", "restless",
        "paranoid", "overthinking", "what if", "nightmare", "phobia",
        "shaking", "heart racing", "can't breathe", "suffocating",
    },
}

# Minimum confidence delta below which we treat the prediction as uncertain
_UNCERTAIN_MARGIN = 0.15


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain(text: str, prediction: dict[str, Any]) -> dict[str, Any]:
    """
    Generate a lightweight explanation for a mental-state prediction.

    Parameters
    ----------
    text : str
        The raw user utterance that was classified.
    prediction : dict
        The dict returned by ``MentalStatePredictor.predict``, expected keys:
            ``label`` (str), ``confidence`` (float), ``scores`` (dict).

    Returns
    -------
    dict with keys:
        ``keywords``        — list[str]  matched emotional keywords
        ``sentiment_score`` — float      polarity in [-1.0, 1.0]
        ``reason``          — str        human-readable explanation paragraph
    """
    label: str = prediction.get("label", "Neutral")
    confidence: float = prediction.get("confidence", 0.0)
    scores: dict[str, float] = prediction.get("scores", {})

    # ── 1. Keyword scan ──────────────────────────────────────────────────────
    tokens = re.findall(r"[a-z']+", text.lower())
    token_set = set(tokens)

    matched_keywords: list[str] = []
    category_hits: dict[str, list[str]] = {}
    for category, kw_set in _KEYWORD_SETS.items():
        hits = sorted(token_set & kw_set)
        # Also scan for multi-word phrases
        for phrase in kw_set:
            if " " in phrase and phrase in text.lower() and phrase not in hits:
                hits.append(phrase)
        if hits:
            category_hits[category] = hits
            matched_keywords.extend(hits)

    matched_keywords = list(dict.fromkeys(matched_keywords))  # dedupe, preserve order

    # ── 2. Sentiment score ───────────────────────────────────────────────────
    sentiment_score = round(_get_sentiment(text), 4)

    # ── 3. Build human-readable reason ───────────────────────────────────────
    reason = _build_reason(
        text=text,
        label=label,
        confidence=confidence,
        scores=scores,
        matched_keywords=matched_keywords,
        category_hits=category_hits,
        sentiment_score=sentiment_score,
    )

    return {
        "keywords": matched_keywords,
        "sentiment_score": sentiment_score,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Reason builder — pure logic, no I/O
# ---------------------------------------------------------------------------

def _build_reason(
    text: str,
    label: str,
    confidence: float,
    scores: dict[str, float],
    matched_keywords: list[str],
    category_hits: dict[str, list[str]],
    sentiment_score: float,
) -> str:
    """Compose a concise, human-readable explanation string."""

    parts: list[str] = []

    # --- Confidence framing ------------------------------------------------
    if confidence >= 0.80:
        certainty = "strongly"
    elif confidence >= 0.60:
        certainty = "moderately"
    else:
        certainty = "tentatively"

    label_lower = label.lower()
    parts.append(
        f"The model {certainty} predicts **{label}** "
        f"({confidence:.0%} confidence)."
    )

    # --- Keyword evidence --------------------------------------------------
    if matched_keywords:
        kw_display = ", ".join(f'"{k}"' for k in matched_keywords[:6])
        cat_names = list(category_hits.keys())

        if len(cat_names) == 1:
            cat_str = cat_names[0]
        elif len(cat_names) == 2:
            cat_str = f"{cat_names[0]} and {cat_names[1]}"
        else:
            cat_str = ", ".join(cat_names[:-1]) + f", and {cat_names[-1]}"

        parts.append(
            f"The message contains language associated with **{cat_str}** "
            f"({kw_display})."
        )
    else:
        parts.append(
            "No strong emotional keywords were detected; "
            "the classification relies primarily on contextual patterns."
        )

    # --- Sentiment framing -------------------------------------------------
    if sentiment_score <= -0.5:
        sent_desc = "strongly negative"
    elif sentiment_score <= -0.1:
        sent_desc = "mildly negative"
    elif sentiment_score >= 0.5:
        sent_desc = "strongly positive"
    elif sentiment_score >= 0.1:
        sent_desc = "mildly positive"
    else:
        sent_desc = "neutral"

    parts.append(
        f"Overall sentiment polarity is **{sent_desc}** "
        f"(score: {sentiment_score:+.2f})."
    )

    # --- Label-specific interpretations ------------------------------------
    if label == "Neutral":
        parts.append(
            "The language pattern suggests a calm, stable emotional state "
            "with no significant distress signals."
        )
    elif label == "Stressed":
        parts.append(
            "Indicators of elevated pressure, urgency, or cognitive overload "
            "were detected, suggesting the person may be under significant stress."
        )
    elif label == "Depressed":
        parts.append(
            "The text reflects themes of low mood, withdrawal, or feelings of "
            "hopelessness — patterns commonly associated with depressive states."
        )

    # --- Runner-up note (when scores are close) ----------------------------
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_scores) >= 2:
        runner_label, runner_score = sorted_scores[1]
        top_score = sorted_scores[0][1]
        if (top_score - runner_score) < _UNCERTAIN_MARGIN:
            parts.append(
                f"Note: **{runner_label}** was a close second "
                f"({runner_score:.0%}), so the result carries some uncertainty."
            )

    return "  \n".join(parts)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    _samples = [
        {
            "text": "I just can't cope anymore. Work is overwhelming and I haven't slept.",
            "pred": {"label": "Stressed", "confidence": 0.88,
                     "scores": {"Neutral": 0.05, "Stressed": 0.88, "Depressed": 0.07}},
        },
        {
            "text": "Everything feels pointless. I don't see the point in going on.",
            "pred": {"label": "Depressed", "confidence": 0.73,
                     "scores": {"Neutral": 0.08, "Stressed": 0.19, "Depressed": 0.73}},
        },
        {
            "text": "Yeah, things are pretty calm today. No complaints.",
            "pred": {"label": "Neutral", "confidence": 0.91,
                     "scores": {"Neutral": 0.91, "Stressed": 0.06, "Depressed": 0.03}},
        },
    ]

    for sample in _samples:
        result = explain(sample["text"], sample["pred"])
        print(f"\nText    : {sample['text']}")
        print(f"Label   : {sample['pred']['label']}")
        print(json.dumps(result, indent=2))
