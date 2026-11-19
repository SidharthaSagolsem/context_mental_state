# Context-Aware Mental State Detector 🧠

A context-aware sequence classification pipeline and interactive Streamlit application that detects mental states (**Neutral**, **Stressed**, **Depressed**) across multi-turn conversational dialogue using fine-tuned transformer models (`distilroberta-base`).

Unlike single-turn sentiment classifiers, this system incorporates prior conversational history using a sliding context window to capture emotional trajectories and subtle mental health signals.

---

## 🌟 Key Features

- **Context-Aware Inference**: Concatenates recent conversational turns using separator tokens (`[SEP]`) to evaluate mental state in context.
- **Explainability & Attribution**: Highlights trigger words, key phrases, and sentiment indicators that contributed to the model's prediction.
- **Interaction Memory & Trend Tracking**: Tracks emotional states across conversational sessions to visualize shifts and progression over time.
- **Interactive Streamlit Interface**: Clean, clinical dashboard with real-time prediction badges, probability distribution charts, and conversation history.
- **In-App & Standalone Training**: Train or fine-tune models directly via the Streamlit sidebar or with the standalone training script.
- **Decoupled Architecture**: Framework-agnostic inference engine (`MentalStatePredictor`) ready for integration into web APIs, bots, or CLI tools.

---

## 📁 Project Structure

```text
context_mental_state/
├── app.py                   # Streamlit web application & interactive UI
├── requirements.txt         # Project dependencies
├── synthetic_dataset.csv    # Multi-turn conversational dataset
├── src/
│   ├── data_generator.py    # Synthetic dialogue generator for mental state scenarios
│   ├── explainability.py    # Linguistic and sentiment explainability engine
│   ├── inference.py         # Framework-agnostic predictor pipeline
│   ├── memory.py            # Long-term session memory & trend analysis
│   ├── preprocess.py        # Dataset tokenizer & context-window formatter
│   └── train.py             # Hugging Face Trainer pipeline (distilroberta-base)
├── saved_model/             # Exported model weights and tokenizer (generated on train)
└── checkpoints/             # Training checkpoints (generated on train)
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10 or higher
- Git

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/SidharthaSagolsem/context_mental_state.git
cd context_mental_state

# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🛠️ Usage

### Training the Model

You can train the model in one of two ways:

1. **Via CLI**:
   ```bash
   python -m src.train
   ```
   This fine-tunes `distilroberta-base` on `synthetic_dataset.csv` and saves the model artifacts to `./saved_model/`.

2. **Via Streamlit UI**:
   Launch the app and click **"Train Model Locally"** in the sidebar.

### Running the Web Application

Start the Streamlit application:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📊 Model & Training Details

- **Base Architecture**: `distilroberta-base`
- **Classes**: `Neutral` (0), `Stressed` (1), `Depressed` (2)
- **Context Window**: 3 recent user dialogue turns
- **Max Token Length**: 128
- **Evaluation Metric**: Macro F1-score & accuracy

---

## 🛡️ Disclaimer

This tool is designed for educational, research, and assistive purposes only. It is **not** a diagnostic medical device or a substitute for professional mental health diagnosis, counseling, or clinical intervention.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
