"""
src/inference.py
================
Decoupled inference pipeline for the Context-Aware Mental State Detector.

``MentalStatePredictor`` is intentionally framework-agnostic:
  • It has no dependency on Streamlit, Flask, FastAPI, or any other serving layer.
  • It can be imported into a REST API, a voice assistant, a CLI tool, or a
    test suite without modification.
  • It mirrors the preprocessing logic from preprocess.py to guarantee that
    the model sees inputs at inference time that are identical in format to
    those it saw during training.

Expected contract
-----------------
  predictor = MentalStatePredictor()          # loads model from ./saved_model/
  result = predictor.predict(
      current_text    = "I can't cope anymore.",
      context_history = [
          "How are things going?",
          "I've been under a lot of pressure.",
      ],
  )
  # → {"label": "Stressed", "confidence": 0.87, "scores": {...}}
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure project root is on the path when running this file directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Constants (must match training configuration)
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR = str(ROOT / "saved_model")
LABEL_NAMES = {0: "Neutral", 1: "Stressed", 2: "Depressed"}
CONTEXT_WINDOW = 3          # Must match the value used during training
MAX_LENGTH = 128             # Must match the value used during training
SEP_PLACEHOLDER = " [SEP] " # Fallback if tokenizer has no sep_token


class MentalStatePredictor:
    """
    Loads a fine-tuned distilroberta-base model and exposes a single ``predict``
    method for context-aware mental state inference.

    Args:
        model_dir : Path to the saved model directory (output of train.py).
                    Defaults to ``./saved_model``.
        device    : PyTorch device string ("cpu", "cuda", "mps").
                    Defaults to GPU if available, else CPU.
    """

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, device: str | None = None):
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(
                f"Model directory '{model_dir}' not found. "
                "Please run the training pipeline first (src/train.py or via the app)."
            )

        # Resolve device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)
        print(f"[inference] Loading model from '{model_dir}' on device '{device}' …")

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        # Determine the separator string (mirrors preprocess.py logic)
        self.sep_string = (
            f" {self.tokenizer.sep_token} "
            if self.tokenizer.sep_token
            else SEP_PLACEHOLDER
        )

        print(f"[inference] Model loaded. Labels: {LABEL_NAMES}")

    # ------------------------------------------------------------------
    # Input formatting (mirrors ConversationDataset.format_input)
    # ------------------------------------------------------------------

    def _format_input(self, current_text: str, context_history: list[str]) -> str:
        """
        Concatenate the last ``CONTEXT_WINDOW`` context turns with the current
        utterance using the separator token.

        Args:
            current_text    : The utterance to classify.
            context_history : List of preceding turns (oldest first).
                              Caller may pass the full history; this method
                              takes only the tail of length CONTEXT_WINDOW.

        Returns:
            A single string ready for tokenization.
        """
        # Take only the last CONTEXT_WINDOW turns to stay within training distribution
        trimmed_context = context_history[-CONTEXT_WINDOW:] if context_history else []
        parts = trimmed_context + [current_text]
        return self.sep_string.join(parts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, current_text: str, context_history: list[str] | None = None) -> dict:
        """
        Predict the mental state for ``current_text`` given conversational context.

        Args:
            current_text    : The most recent utterance to classify.
            context_history : Ordered list of preceding utterances (oldest first).
                              Pass an empty list or ``None`` for isolated utterances.

        Returns:
            Dict with keys:
                "label"      : str  — predicted class name ("Neutral" | "Stressed" | "Depressed")
                "label_id"   : int  — integer class index (0 | 1 | 2)
                "confidence" : float — probability of the predicted class (0.0–1.0)
                "scores"     : dict — {label_name: probability} for all classes
        """
        if context_history is None:
            context_history = []

        # 1. Format input string (identical to training preprocessing)
        formatted_input = self._format_input(current_text, context_history)

        # 2. Tokenize
        encoding = self.tokenizer(
            formatted_input,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        # 3. Forward pass (no gradient computation required at inference)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # shape: (1, num_labels)

        # 4. Convert logits → probabilities via softmax
        probs = F.softmax(logits, dim=-1).squeeze(0)  # shape: (num_labels,)

        # 5. Extract predicted class
        predicted_id = int(torch.argmax(probs).item())
        confidence = float(probs[predicted_id].item())
        label_name = LABEL_NAMES[predicted_id]

        # Build per-class score dict
        scores = {
            LABEL_NAMES[i]: round(float(probs[i].item()), 4)
            for i in range(len(LABEL_NAMES))
        }

        return {
            "label": label_name,
            "label_id": predicted_id,
            "confidence": round(confidence, 4),
            "scores": scores,
        }

    def is_ready(self) -> bool:
        """Return True if the model is loaded and ready for inference."""
        return self.model is not None


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    predictor = MentalStatePredictor()

    test_cases = [
        {
            "context": ["How was your day?", "Pretty quiet, nothing much happening."],
            "current": "Yeah, everything is fine I suppose.",
        },
        {
            "context": ["How are things at work?", "Kind of hectic lately."],
            "current": "I have four deadlines tomorrow and I haven't slept properly in days.",
        },
        {
            "context": ["What have you been up to?", "Not much really."],
            "current": "I don't really see the point in anything anymore. Everything feels grey.",
        },
    ]

    print("\n=== MentalStatePredictor Smoke Test ===\n")
    for i, tc in enumerate(test_cases, 1):
        result = predictor.predict(
            current_text=tc["current"],
            context_history=tc["context"],
        )
        print(f"Test {i}")
        print(f"  Context : {tc['context']}")
        print(f"  Utterance: {tc['current']}")
        print(f"  → Label : {result['label']}  (confidence: {result['confidence']:.2%})")
        print(f"  → Scores: {result['scores']}")
        print()
