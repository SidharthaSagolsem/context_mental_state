"""
src/train.py
============
Training pipeline for the Context-Aware Mental State Detector.

Uses the Hugging Face ``Trainer`` API with ``distilroberta-base`` fine-tuned
for 3-class sequence classification (Neutral / Stressed / Depressed).

Usage (standalone):
    python -m src.train

The trained model and tokenizer are saved to ``./saved_model/``.
"""

import os
import sys
import numpy as np
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# Ensure the project root is on sys.path when run directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_generator import generate_dataset, LABEL_NAMES
from src.preprocess import build_datasets

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "distilroberta-base"
NUM_LABELS = 3
SAVED_MODEL_DIR = str(ROOT / "saved_model")
DATA_OUTPUT_PATH = str(ROOT / "synthetic_dataset.csv")

# Training hyper-parameters — kept small for local execution
TRAINING_CONFIG = {
    "output_dir": str(ROOT / "checkpoints"),
    "num_train_epochs": 10,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 16,
    "warmup_steps": 50,
    "weight_decay": 0.01,
    "logging_dir": str(ROOT / "logs"),
    "logging_steps": 10,
    "eval_strategy": "epoch",  # evaluate after every epoch
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1_macro",
    "greater_is_better": True,
    "report_to": "none",             # disable wandb / tensorboard for simplicity
    "seed": 42,
}

# Label-to-id / id-to-label maps required by the HF model config
LABEL2ID = {name: idx for idx, name in LABEL_NAMES.items()}
ID2LABEL = {idx: name for idx, name in LABEL_NAMES.items()}


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred) -> dict:
    """
    Compute accuracy, macro-F1, macro-precision, and macro-recall.

    This function is called by the Trainer after each evaluation step.

    Args:
        eval_pred: NamedTuple with fields ``predictions`` (logits) and
                   ``label_ids`` (ground-truth integer labels).

    Returns:
        Dict of metric names → float values.
    """
    logits, labels = eval_pred
    # Convert raw logits to class indices
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    prec = precision_score(labels, predictions, average="macro", zero_division=0)
    rec = recall_score(labels, predictions, average="macro", zero_division=0)

    return {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1, 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    episodes_per_class: int = 40,
    turns_per_episode: int = 4,
    epochs: int = 10,
    batch_size: int = 8,
    progress_callback=None,
) -> str:
    """
    Full end-to-end training pipeline.

    Args:
        episodes_per_class : Episodes per label class for data generation.
        turns_per_episode  : Turns per episode (affects dataset size).
        epochs             : Number of training epochs.
        batch_size         : Per-device training batch size.
        progress_callback  : Optional callable(message: str) for UI updates.

    Returns:
        Path to the saved model directory.
    """

    def log(msg: str):
        print(f"[train] {msg}", flush=True)
        if progress_callback:
            progress_callback(msg)

    # ------------------------------------------------------------------
    # Step 1 — Generate synthetic data
    # ------------------------------------------------------------------
    log("Generating synthetic conversational dataset …")
    df = generate_dataset(
        episodes_per_class=episodes_per_class,
        turns_per_episode=turns_per_episode,
        output_path=DATA_OUTPUT_PATH,
    )
    log(f"Dataset generated: {len(df)} total rows, "
        f"{df['episode_id'].nunique()} episodes.")

    # ------------------------------------------------------------------
    # Step 2 — Tokenize and build PyTorch datasets
    # ------------------------------------------------------------------
    log(f"Building datasets with tokenizer '{MODEL_NAME}' …")
    train_ds, val_ds, tokenizer = build_datasets(
        df=df,
        tokenizer_name=MODEL_NAME,
        max_length=128,
        context_window=3,
        val_ratio=0.15,
    )

    # ------------------------------------------------------------------
    # Step 3 — Initialise the model
    # ------------------------------------------------------------------
    log(f"Loading '{MODEL_NAME}' for sequence classification …")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # ------------------------------------------------------------------
    # Step 4 — Configure training arguments
    # ------------------------------------------------------------------
    training_args_dict = TRAINING_CONFIG.copy()
    training_args_dict["num_train_epochs"] = epochs
    training_args_dict["per_device_train_batch_size"] = batch_size

    training_args = TrainingArguments(**training_args_dict)

    # ------------------------------------------------------------------
    # Step 5 — Instantiate Trainer and run
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    log("Starting training … (this may take a few minutes on CPU)")
    trainer.train()

    # ------------------------------------------------------------------
    # Step 6 — Final evaluation
    # ------------------------------------------------------------------
    log("Running final evaluation …")
    metrics = trainer.evaluate()
    log(f"Evaluation results: {metrics}")

    # ------------------------------------------------------------------
    # Step 7 — Save the best model + tokenizer
    # ------------------------------------------------------------------
    os.makedirs(SAVED_MODEL_DIR, exist_ok=True)
    log(f"Saving model to '{SAVED_MODEL_DIR}' …")
    trainer.save_model(SAVED_MODEL_DIR)
    tokenizer.save_pretrained(SAVED_MODEL_DIR)
    log("Training complete. Model saved successfully.")

    return SAVED_MODEL_DIR


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    saved_path = train()
    print(f"\n✓ Model saved to: {saved_path}")
