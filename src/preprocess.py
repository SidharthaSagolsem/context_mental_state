"""
src/preprocess.py
=================
Defines the PyTorch Dataset class for context-aware mental state detection.

Core idea
---------
For each USER turn in the dataset, we look back up to ``context_window`` turns
within the SAME episode and concatenate them as:

    ctx_n-2 [SEP] ctx_n-1 [SEP] current_utterance

This gives the model conversational history so it can detect gradual mood shifts,
not just isolated utterances.

Only USER turns (label != -1) are returned as training examples.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Label map (mirrors data_generator.py)
# ---------------------------------------------------------------------------
LABEL_NAMES = {0: "Neutral", 1: "Stressed", 2: "Depressed"}
ID_TO_LABEL = {k: v for k, v in LABEL_NAMES.items()}


class ConversationDataset(Dataset):
    """
    PyTorch Dataset that yields (input_ids, attention_mask, label) tensors.

    For each USER utterance the class:
      1. Identifies up to ``context_window`` preceding turns in the same episode.
      2. Concatenates them with the tokenizer's ``sep_token`` in between.
      3. Tokenizes the resulting string with padding / truncation.

    Args:
        df              : Raw DataFrame from data_generator.generate_dataset().
        tokenizer       : A Hugging Face AutoTokenizer instance.
        max_length      : Maximum token length (padding / truncation target).
        context_window  : How many preceding turns to include (default 3).
    """

    SEP_PLACEHOLDER = " [SEP] "  # Used when tokenizer has no explicit sep_token

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        context_window: int = 3,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.context_window = context_window

        # Choose the separator string based on the tokenizer vocabulary
        self.sep_string = (
            f" {tokenizer.sep_token} "
            if tokenizer.sep_token
            else self.SEP_PLACEHOLDER
        )

        # Build a row-level text lookup (row_index -> text) so _get_context_texts
        # can resolve DataFrame indices without holding a reference to the full df.
        self._row_text_lookup: dict = df["text"].to_dict()

        # Pre-build grouped index: episode_id -> sorted list of (turn_id, row_index)
        # This lets __getitem__ quickly fetch preceding turns.
        self._episode_turns: dict[int, list[tuple[int, int]]] = {}
        for row_idx, row in df.iterrows():
            ep = row["episode_id"]
            self._episode_turns.setdefault(ep, []).append((row["turn_id"], row_idx))
        # Sort each episode by turn_id to maintain chronological order
        for ep in self._episode_turns:
            self._episode_turns[ep].sort(key=lambda x: x[0])

        # Keep only USER turns as training examples (bot turns are not labelled)
        self.samples: list[dict] = []
        for _, row in df[df["label"] != -1].iterrows():
            self.samples.append(
                {
                    "episode_id": row["episode_id"],
                    "turn_id": int(row["turn_id"]),
                    "text": str(row["text"]),
                    "label": int(row["label"]),
                }
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_context_texts(self, episode_id: int, current_turn_id: int) -> list[str]:
        """
        Return up to ``context_window`` utterances that precede the current turn
        in the same episode, in chronological order.

        Args:
            episode_id      : Episode the current turn belongs to.
            current_turn_id : turn_id of the current utterance (excluded).

        Returns:
            List of text strings (oldest first).
        """
        # Retrieve all turns for this episode (already sorted)
        all_turns = self._episode_turns.get(episode_id, [])
        # Grab texts of turns that came BEFORE the current turn
        preceding = [
            (tid, idx) for (tid, idx) in all_turns if tid < current_turn_id
        ]
        # Take only the last `context_window` turns
        preceding = preceding[-self.context_window :]
        return [self._df_row_text(idx) for (_, idx) in preceding]

    def _df_row_text(self, row_idx) -> str:
        """Helper: retrieve the text field for a given DataFrame row index."""
        # We stored samples in a list; retrieve from the original grouped index.
        # Since _episode_turns stores DataFrame indices, we need the original df.
        # We'll resolve via the _row_lookup dict built in __init__.
        return self._row_text_lookup[row_idx]

    # ------------------------------------------------------------------
    # Public helpers (also used by inference.py for consistent formatting)
    # ------------------------------------------------------------------

    @staticmethod
    def format_input(current_text: str, context_texts: list[str], sep_string: str) -> str:
        """
        Concatenate context turns and the current utterance into a single string.

        Example output (with 2 context turns):
            "Hello [SEP] How are you? [SEP] I'm feeling terrible."

        Args:
            current_text   : The utterance to classify.
            context_texts  : List of preceding utterances (oldest first).
            sep_string     : Separator token string.

        Returns:
            Formatted string ready for tokenization.
        """
        parts = context_texts + [current_text]
        return sep_string.join(parts)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns a dict with keys: input_ids, attention_mask, labels.
        The Hugging Face Trainer expects the label key to be ``labels``.
        """
        sample = self.samples[idx]

        # Retrieve preceding context
        context_texts = self._get_context_texts(
            episode_id=sample["episode_id"],
            current_turn_id=sample["turn_id"],
        )

        # Build the concatenated input string
        full_input = self.format_input(
            current_text=sample["text"],
            context_texts=context_texts,
            sep_string=self.sep_string,
        )

        # Tokenize
        encoding = self.tokenizer(
            full_input,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),       # (seq_len,)
            "attention_mask": encoding["attention_mask"].squeeze(0),  # (seq_len,)
            "labels": torch.tensor(sample["label"], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_datasets(
    df: pd.DataFrame,
    tokenizer_name: str = "distilroberta-base",
    max_length: int = 128,
    context_window: int = 3,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple["ConversationDataset", "ConversationDataset", AutoTokenizer]:
    """
    Convenience factory: loads the tokenizer, splits the DataFrame, and
    returns (train_dataset, val_dataset, tokenizer).

    Args:
        df             : Full dataset DataFrame.
        tokenizer_name : Hugging Face model name (default: distilroberta-base).
        max_length     : Token sequence length.
        context_window : Number of context turns.
        val_ratio      : Fraction of episodes to use for validation.
        seed           : Random seed for reproducible splits.

    Returns:
        (train_ds, val_ds, tokenizer)
    """
    import random

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Split by episode_id to avoid data leakage across turns of the same episode
    all_episode_ids = df["episode_id"].unique().tolist()
    random.seed(seed)
    random.shuffle(all_episode_ids)

    split_point = int(len(all_episode_ids) * (1 - val_ratio))
    train_episodes = set(all_episode_ids[:split_point])
    val_episodes = set(all_episode_ids[split_point:])

    train_df = df[df["episode_id"].isin(train_episodes)].reset_index(drop=True)
    val_df = df[df["episode_id"].isin(val_episodes)].reset_index(drop=True)

    # _row_text_lookup is now built inside ConversationDataset.__init__ directly
    # from the df passed in, so no external attachment is needed.
    train_ds = ConversationDataset(train_df, tokenizer, max_length, context_window)
    val_ds = ConversationDataset(val_df, tokenizer, max_length, context_window)

    print(
        f"[preprocess] Train samples: {len(train_ds)} | "
        f"Val samples: {len(val_ds)} | "
        f"Tokenizer: {tokenizer_name}"
    )
    return train_ds, val_ds, tokenizer
