"""
src/memory.py
=============
Lightweight emotional-history memory for the Context-Aware Mental State Detector.

Two responsibilities
--------------------
1. **InteractionMemory** — append-only log of recent predictions held in a
   plain Python list (lives inside ``st.session_state``; zero I/O, zero deps).
2. **analyze_trend()** — pure-function trend analyser that reads the log and
   returns a structured insight dict ready for the UI.

Design choices
--------------
* No database, no files — the list is stored in Streamlit session state.
* All public functions are pure or nearly-pure; easy to unit-test.
* Dependency-free: only the Python standard library is used.

Usage (agent integration)
--------------------------
    from src.memory import InteractionMemory, analyze_trend

    # After prediction + explanation are ready:
    memory.add(text, prediction, explanation)

    # Build trend insight:
    trend = analyze_trend(memory.entries)
    # → {"trend": "worsening", "confidence": 0.72, "summary": "…"}
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_ENTRIES = 20          # rolling window size
MIN_ENTRIES_FOR_TREND = 3 # need at least this many to compute a meaningful trend

_LABEL_WEIGHTS = {
    "Neutral":   0.0,   # neutral baseline
    "Stressed":  0.5,   # moderate distress
    "Depressed": 1.0,   # high distress
}


# ---------------------------------------------------------------------------
# InteractionMemory
# ---------------------------------------------------------------------------

class InteractionMemory:
    """
    Append-only rolling log of recent agent interactions.

    Each entry is a plain dict:
    {
        "timestamp":       str (ISO-8601 UTC),
        "text":            str,
        "label":           str,
        "confidence":      float,
        "sentiment_score": float | None,
        "distress_weight": float,   # derived from label
    }

    Keeps at most ``max_entries`` entries (oldest discarded on overflow).
    """

    def __init__(self, max_entries: int = MAX_ENTRIES):
        self.max_entries = max_entries
        self.entries: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        prediction: dict[str, Any],
        explanation: dict[str, Any] | None = None,
    ) -> None:
        """
        Append a new interaction to the memory.

        Parameters
        ----------
        text       : Raw user utterance.
        prediction : Dict returned by ``MentalStatePredictor.predict``.
        explanation: Dict returned by ``explainability.explain`` (optional).
        """
        label      = prediction.get("label", "Neutral")
        confidence = prediction.get("confidence", 0.0)
        sentiment  = (
            explanation.get("sentiment_score")
            if explanation
            else None
        )

        entry: dict[str, Any] = {
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "text":            text,
            "label":           label,
            "confidence":      confidence,
            "sentiment_score": sentiment,
            "distress_weight": _LABEL_WEIGHTS.get(label, 0.0),
        }

        self.entries.append(entry)

        # Trim to rolling window
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

    def clear(self) -> None:
        """Reset the memory (e.g., when the user clears the conversation)."""
        self.entries = []

    def __len__(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def sentiment_series(self) -> list[float]:
        """Return non-None sentiment scores in chronological order."""
        return [
            e["sentiment_score"]
            for e in self.entries
            if e["sentiment_score"] is not None
        ]

    def distress_series(self) -> list[float]:
        """Return distress weights in chronological order."""
        return [e["distress_weight"] for e in self.entries]

    def label_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {"Neutral": 0, "Stressed": 0, "Depressed": 0}
        for e in self.entries:
            counts[e["label"]] = counts.get(e["label"], 0) + 1
        return counts


# ---------------------------------------------------------------------------
# Trend analysis — pure function
# ---------------------------------------------------------------------------

def analyze_trend(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute an emotional trend from a list of memory entries.

    Parameters
    ----------
    entries : List of entry dicts as produced by ``InteractionMemory``.
              Typically ``memory.entries``.

    Returns
    -------
    dict with keys:
        ``trend``      — "improving" | "worsening" | "stable"
        ``confidence`` — float in [0, 1] (how certain the trend call is)
        ``summary``    — human-readable explanation string
        ``stats``      — raw numeric stats (for charting / debugging)
    """
    n = len(entries)

    if n < MIN_ENTRIES_FOR_TREND:
        return {
            "trend":      "stable",
            "confidence": 0.0,
            "summary":    (
                f"Not enough data yet — {n} of {MIN_ENTRIES_FOR_TREND} "
                "interactions needed to compute a trend."
            ),
            "stats": {},
        }

    # ── Distress series ──────────────────────────────────────────────────────
    distress = [e["distress_weight"] for e in entries]
    avg_distress  = sum(distress) / n

    # Slope via simple linear regression on distress
    distress_slope = _linear_slope(distress)

    # ── Sentiment series ─────────────────────────────────────────────────────
    sentiments = [
        e["sentiment_score"] for e in entries if e["sentiment_score"] is not None
    ]
    avg_sentiment  = sum(sentiments) / len(sentiments) if sentiments else None
    sentiment_slope = _linear_slope(sentiments) if len(sentiments) >= 2 else 0.0

    # ── Label frequency ──────────────────────────────────────────────────────
    label_counts: dict[str, int] = {}
    for e in entries:
        label_counts[e["label"]] = label_counts.get(e["label"], 0) + 1

    negative_rate = (
        label_counts.get("Stressed", 0) + label_counts.get("Depressed", 0)
    ) / n

    depressed_rate = label_counts.get("Depressed", 0) / n

    # ── Trend decision ───────────────────────────────────────────────────────
    # Primary signal: distress slope (positive = getting worse)
    # Corroborates with sentiment slope (negative = getting worse)
    # We blend both signals with equal weight.
    SLOPE_THRESHOLD = 0.04   # minimum slope magnitude to call a trend

    # Normalise slopes to [-1, 1] roughly
    norm_distress_slope  = math.tanh(distress_slope  * 5)
    norm_sentiment_slope = math.tanh(-sentiment_slope * 5)  # flipped: neg sent = worse
    combined = (norm_distress_slope + norm_sentiment_slope) / 2

    if combined > SLOPE_THRESHOLD:
        trend = "worsening"
    elif combined < -SLOPE_THRESHOLD:
        trend = "improving"
    else:
        trend = "stable"

    # Confidence: how strong is the signal?
    raw_conf = min(abs(combined) * 2, 1.0)
    # Dampen confidence when we have very few entries
    damping  = min((n - MIN_ENTRIES_FOR_TREND + 1) / 5, 1.0)
    confidence = round(raw_conf * damping, 3)

    # ── Summary text ─────────────────────────────────────────────────────────
    summary = _build_summary(
        trend=trend,
        n=n,
        avg_distress=avg_distress,
        distress_slope=distress_slope,
        avg_sentiment=avg_sentiment,
        sentiment_slope=sentiment_slope,
        negative_rate=negative_rate,
        depressed_rate=depressed_rate,
        label_counts=label_counts,
    )

    stats: dict[str, Any] = {
        "n":               n,
        "avg_distress":    round(avg_distress, 3),
        "distress_slope":  round(distress_slope, 4),
        "avg_sentiment":   round(avg_sentiment, 3) if avg_sentiment is not None else None,
        "sentiment_slope": round(sentiment_slope, 4),
        "negative_rate":   round(negative_rate, 3),
        "depressed_rate":  round(depressed_rate, 3),
        "label_counts":    label_counts,
        "combined_signal": round(combined, 4),
    }

    return {
        "trend":      trend,
        "confidence": confidence,
        "summary":    summary,
        "stats":      stats,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _linear_slope(values: list[float]) -> float:
    """
    Return the slope (m) of the OLS line y = m*x + b fitted to ``values``
    (where x is the index 0, 1, 2, …).  Returns 0.0 for fewer than 2 points.
    """
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    num   = sum((xs[i] - x_mean) * (values[i] - y_mean) for i in range(n))
    denom = sum((xs[i] - x_mean) ** 2 for i in range(n))
    return num / denom if denom else 0.0


def _build_summary(
    trend: str,
    n: int,
    avg_distress: float,
    distress_slope: float,
    avg_sentiment: float | None,
    sentiment_slope: float,
    negative_rate: float,
    depressed_rate: float,
    label_counts: dict[str, int],
) -> str:
    """Compose a concise, human-readable trend summary."""

    parts: list[str] = []

    # ── Trend framing ───────────────────────────────────────────────────────
    window = f"last {n} interaction{'s' if n != 1 else ''}"
    if trend == "worsening":
        parts.append(
            f"⚠️ Emotional state has been **worsening** over the {window}."
        )
    elif trend == "improving":
        parts.append(
            f"✅ Emotional state appears to be **improving** over the {window}."
        )
    else:
        parts.append(
            f"📊 Emotional state has been broadly **stable** over the {window}."
        )

    # ── Distress detail ─────────────────────────────────────────────────────
    if distress_slope > 0.05:
        parts.append("Distress indicators are trending **upward**.")
    elif distress_slope < -0.05:
        parts.append("Distress indicators are trending **downward**.")

    # ── Sentiment detail ────────────────────────────────────────────────────
    if avg_sentiment is not None:
        if avg_sentiment <= -0.3:
            parts.append(
                f"Average sentiment is **strongly negative** ({avg_sentiment:+.2f})."
            )
        elif avg_sentiment <= -0.05:
            parts.append(
                f"Average sentiment is **slightly negative** ({avg_sentiment:+.2f})."
            )
        elif avg_sentiment >= 0.3:
            parts.append(
                f"Average sentiment is **positive** ({avg_sentiment:+.2f})."
            )

        if sentiment_slope < -0.03:
            parts.append("Sentiment polarity has been **decreasing** over time.")
        elif sentiment_slope > 0.03:
            parts.append("Sentiment polarity has been **increasing** over time.")

    # ── Label frequency notes ───────────────────────────────────────────────
    if depressed_rate >= 0.5:
        parts.append(
            f"**{depressed_rate:.0%}** of recent messages show depressive signals — "
            "consider reaching out to a professional."
        )
    elif negative_rate >= 0.6:
        parts.append(
            f"**{negative_rate:.0%}** of recent messages contain stress or "
            "depressive language."
        )

    dominant = max(label_counts, key=label_counts.get)  # type: ignore[arg-type]
    parts.append(
        f"Most frequent state: **{dominant}** "
        f"({label_counts[dominant]}/{n} messages)."
    )

    return "  \n".join(parts)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    mem = InteractionMemory(max_entries=20)

    _test_data = [
        ("Things are fine today.",
         {"label": "Neutral",   "confidence": 0.91},
         {"sentiment_score":  0.72}),
        ("Work is a bit hectic.",
         {"label": "Stressed",  "confidence": 0.74},
         {"sentiment_score": -0.20}),
        ("I'm really overwhelmed right now.",
         {"label": "Stressed",  "confidence": 0.88},
         {"sentiment_score": -0.45}),
        ("I don't see a way out. Everything feels pointless.",
         {"label": "Depressed", "confidence": 0.83},
         {"sentiment_score": -0.68}),
        ("I just can't cope anymore.",
         {"label": "Depressed", "confidence": 0.79},
         {"sentiment_score": -0.61}),
    ]

    for text, pred, expl in _test_data:
        mem.add(text, pred, expl)

    trend = analyze_trend(mem.entries)
    print("Memory entries:", len(mem))
    print(json.dumps(trend, indent=2))
