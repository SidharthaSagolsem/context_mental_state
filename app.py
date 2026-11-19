"""
app.py
======
Streamlit web application for the Context-Aware Mental State Detector.

Features:
  • Real-time mental state prediction as the user chats.
  • Session-state-backed chat history (context window persists across turns).
  • Colour-coded mental state badge (green / amber / red).
  • Confidence score and per-class probability bar chart.
  • Sidebar "Train Model Locally" button that runs the full training pipeline
    in-process so the user never has to leave the UI.

Run:
    streamlit run app.py
"""

import sys
import os
import threading
from pathlib import Path

import streamlit as st
from src.explainability import explain as _explain_prediction
from src.memory import InteractionMemory, analyze_trend as _analyze_trend

# ---------------------------------------------------------------------------
# Page config — must be the FIRST Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Mental State Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

SAVED_MODEL_DIR = str(ROOT / "saved_model")

# ---------------------------------------------------------------------------
# Custom CSS — clean, clinical aesthetic with clear state colour coding
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ---- Global font ---- */
    html, body, [class*="css"], .stMarkdown, .stTextInput input,
    .stButton > button, .stSlider, label, p, span, div {
        font-family: 'Inter', sans-serif !important;
    }

    /* ---- Animated gradient background ---- */
    .stApp {
        background: linear-gradient(135deg, #05060f 0%, #0d1021 40%, #0f1525 70%, #060910 100%);
        background-size: 400% 400%;
        animation: gradientShift 18s ease infinite;
        min-height: 100vh;
    }

    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ---- Block container ---- */
    .block-container {
        padding: 2rem 2.5rem !important;
        max-width: 1400px;
        position: relative;
        z-index: 1;
    }

    /* ---- Page title / headers ---- */
    h1, h2, h3 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        color: #f0f4ff !important;
        letter-spacing: -0.02em !important;
    }

    h2 { font-size: 1.45rem !important; }
    h3 { font-size: 1.1rem !important; }

    /* ---- Sidebar chrome ---- */
    [data-testid="stSidebar"] {
        background: rgba(8, 10, 22, 0.92) !important;
        backdrop-filter: blur(24px) !important;
        border-right: 1px solid rgba(99,102,241,0.15) !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #cbd5e1;
    }

    /* ---- Sidebar brand header ---- */
    [data-testid="stSidebar"] h2 {
        background: linear-gradient(90deg, #818cf8, #a78bfa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 1.25rem !important;
        margin-bottom: 0.5rem;
    }

    /* ---- Dividers ---- */
    hr {
        border: none !important;
        border-top: 1px solid rgba(255,255,255,0.08) !important;
        margin: 1rem 0 !important;
    }

    /* ---- Buttons — primary ---- */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.6rem 1.2rem !important;
        letter-spacing: 0.02em !important;
        transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(99,102,241,0.5) !important;
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 100%) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
        box-shadow: 0 2px 10px rgba(99,102,241,0.25) !important;
    }

    .stButton > button:disabled {
        background: rgba(99,102,241,0.25) !important;
        color: rgba(255,255,255,0.4) !important;
        box-shadow: none !important;
        transform: none !important;
        cursor: not-allowed !important;
    }

    /* ---- Form submit button  ---- */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.65rem 1.4rem !important;
        width: 100% !important;
        transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.3) !important;
        letter-spacing: 0.03em !important;
    }

    .stFormSubmitButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(99,102,241,0.5) !important;
    }

    /* ---- Text input ---- */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(99,102,241,0.25) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-size: 0.95rem !important;
        padding: 0.75rem 1rem !important;
        caret-color: #818cf8;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: rgba(99,102,241,0.6) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
        background: rgba(255,255,255,0.07) !important;
        outline: none !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #475569 !important;
    }

    /* ---- Slider ---- */
    .stSlider [data-testid="stTickBar"] { display: none; }
    .stSlider [data-baseweb="slider"] > div > div {
        background: rgba(99,102,241,0.2) !important;
    }
    .stSlider [data-baseweb="slider"] > div > div > div:last-child {
        background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
    }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #818cf8 !important;
        border: 2px solid #fff !important;
        width: 18px !important;
        height: 18px !important;
        box-shadow: 0 0 12px rgba(99,102,241,0.5) !important;
    }

    /* ---- Chat bubbles ---- */
    .user-bubble {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: #fff;
        padding: 13px 20px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0 10px 18%;
        font-size: 0.94rem;
        line-height: 1.6;
        box-shadow: 0 6px 24px rgba(99,102,241,0.35);
        animation: bubbleIn 0.3s cubic-bezier(0.34,1.56,0.64,1) both;
        position: relative;
    }

    .user-bubble::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: inherit;
        background: linear-gradient(135deg, rgba(255,255,255,0.08), transparent);
        pointer-events: none;
    }

    .bot-bubble {
        background: rgba(255,255,255,0.055);
        backdrop-filter: blur(12px);
        color: #d1d9ef;
        padding: 13px 20px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 18% 10px 0;
        font-size: 0.94rem;
        line-height: 1.6;
        border: 1px solid rgba(255,255,255,0.09);
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        animation: bubbleIn 0.3s cubic-bezier(0.34,1.56,0.64,1) both;
    }

    @keyframes bubbleIn {
        from { opacity: 0; transform: translateY(10px) scale(0.97); }
        to   { opacity: 1; transform: translateY(0) scale(1); }
    }

    /* ---- Mental state badges ---- */
    .badge-neutral {
        background: linear-gradient(135deg, #059669, #10b981);
        color: #fff;
        padding: 7px 18px;
        border-radius: 100px;
        font-weight: 600;
        font-size: 0.875rem;
        letter-spacing: 0.06em;
        display: inline-block;
        box-shadow: 0 0 16px rgba(16,185,129,0.4);
        text-transform: uppercase;
    }

    .badge-stressed {
        background: linear-gradient(135deg, #d97706, #f59e0b);
        color: #fff;
        padding: 7px 18px;
        border-radius: 100px;
        font-weight: 600;
        font-size: 0.875rem;
        letter-spacing: 0.06em;
        display: inline-block;
        box-shadow: 0 0 16px rgba(245,158,11,0.4);
        text-transform: uppercase;
    }

    .badge-depressed {
        background: linear-gradient(135deg, #dc2626, #ef4444);
        color: #fff;
        padding: 7px 18px;
        border-radius: 100px;
        font-weight: 600;
        font-size: 0.875rem;
        letter-spacing: 0.06em;
        display: inline-block;
        box-shadow: 0 0 16px rgba(239,68,68,0.4);
        text-transform: uppercase;
    }

    /* ---- Training log terminal ---- */
    .train-log {
        background: rgba(0, 0, 0, 0.55);
        border-left: 3px solid #6366f1;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 0.77rem;
        color: #7dd3fc;
        max-height: 240px;
        overflow-y: auto;
        line-height: 1.7;
        box-shadow: inset 0 0 30px rgba(0,0,0,0.3);
        scrollbar-width: thin;
        scrollbar-color: rgba(99,102,241,0.3) transparent;
    }

    .train-log::-webkit-scrollbar { width: 4px; }
    .train-log::-webkit-scrollbar-track { background: transparent; }
    .train-log::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.35); border-radius: 4px; }

    /* ---- Probability bars ---- */
    .prob-bar-bg {
        background: rgba(255,255,255,0.07);
        border-radius: 100px;
        height: 7px;
        margin: 5px 0 14px;
        overflow: hidden;
    }

    /* ---- Info callout ---- */
    .info-callout {
        background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(139,92,246,0.08));
        border: 1px solid rgba(99,102,241,0.28);
        border-radius: 14px;
        padding: 14px 20px;
        color: #a5b4fc;
        font-size: 0.87rem;
        line-height: 1.65;
        margin-bottom: 1.25rem;
        backdrop-filter: blur(8px);
    }

    /* ---- Spinner ---- */
    .stSpinner > div { border-top-color: #818cf8 !important; }
    .stAlert {
        border-radius: 12px !important;
        backdrop-filter: blur(8px) !important;
    }

    /* ---- Hide Streamlit default branding ---- */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
    header    { visibility: hidden; }

    /* ---- Scrollbar for main area ---- */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 6px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.55); }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper: load the predictor (cached so it's not reloaded on every rerun)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model …")
def load_predictor():
    """Load MentalStatePredictor from the saved model directory."""
    from src.inference import MentalStatePredictor
    return MentalStatePredictor(model_dir=SAVED_MODEL_DIR)


def model_is_trained() -> bool:
    """Return True if a saved model directory exists and contains the config."""
    return os.path.isfile(os.path.join(SAVED_MODEL_DIR, "config.json"))


# ---------------------------------------------------------------------------
# Helper: render a single mental state result card
# ---------------------------------------------------------------------------

def render_prediction_card(result: dict, explanation: dict | None = None):
    """Render the mental state prediction + optional explanation as a styled card."""
    label = result["label"]
    confidence = result["confidence"]
    scores = result["scores"]

    badge_class = {
        "Neutral": "badge-neutral",
        "Stressed": "badge-stressed",
        "Depressed": "badge-depressed",
    }.get(label, "badge-neutral")

    emoji = {"Neutral": "🟢", "Stressed": "🟠", "Depressed": "🔴"}.get(label, "⚪")

    st.markdown(
        f"""
        <div style="background:rgba(255,255,255,0.04); border-radius:14px;
                    padding:16px 20px; border:1px solid rgba(255,255,255,0.08);
                    margin-top:8px;">
            <div style="font-size:0.75rem; color:#64748b; letter-spacing:0.08em;
                        text-transform:uppercase; margin-bottom:8px;">
                Detected Mental State
            </div>
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:14px;">
                <span style="font-size:1.5rem;">{emoji}</span>
                <span class="{badge_class}">{label}</span>
                <span style="color:#64748b; font-size:0.85rem; margin-left:auto;">
                    {confidence:.1%} confidence
                </span>
            </div>
        """,
        unsafe_allow_html=True,
    )

    # Per-class probability bars
    bar_colors = {
        "Neutral": "#059669",
        "Stressed": "#d97706",
        "Depressed": "#dc2626",
    }
    for cls, prob in scores.items():
        color = bar_colors.get(cls, "#64748b")
        pct = prob * 100
        st.markdown(
            f"""
            <div style="font-size:0.78rem; color:#94a3b8; margin-bottom:2px;">
                {cls} <span style="float:right; font-family:'DM Mono',monospace;">{prob:.2%}</span>
            </div>
            <div class="prob-bar-bg">
                <div style="width:{pct}%; height:100%; background:{color};
                            border-radius:4px; transition:width 0.4s ease;"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Explanation panel (rendered only when explanation is available) ───────
    if explanation:
        keywords = explanation.get("keywords", [])
        sentiment_score = explanation.get("sentiment_score", 0.0)
        reason = explanation.get("reason", "")

        # Sentiment colour
        if sentiment_score <= -0.1:
            sent_color = "#ef4444"
        elif sentiment_score >= 0.1:
            sent_color = "#10b981"
        else:
            sent_color = "#94a3b8"

        kw_html = (
            " ".join(
                f'<span style="background:rgba(99,102,241,0.18); color:#a5b4fc; '
                f'border:1px solid rgba(99,102,241,0.3); border-radius:100px; '
                f'font-size:0.73rem; padding:2px 10px; margin:2px; display:inline-block;">'
                f'{kw}</span>'
                for kw in keywords
            )
            if keywords
            else '<span style="color:#475569; font-size:0.82rem;">none detected</span>'
        )

        # Convert lightweight markdown bold (**text**) to HTML
        import re as _re
        reason_html = _re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", reason)
        reason_html = reason_html.replace("  \n", "<br>")

        # Build HTML via concatenation — never interpolate HTML-containing
        # variables inside an f-string, Streamlit sanitises them.
        explanation_html = (
            '<div style="margin-top:16px; background:rgba(99,102,241,0.07);'
            ' border:1px solid rgba(99,102,241,0.22); border-radius:14px;'
            ' padding:16px 20px;">'

            # ── Header ───────────────────────────────────────────────────────
            '<div style="font-size:0.75rem; color:#818cf8; letter-spacing:0.08em;'
            ' text-transform:uppercase; margin-bottom:12px; font-weight:600;">'
            '🔍 Why this prediction'
            '</div>'

            # ── Keywords ─────────────────────────────────────────────────────
            '<div style="margin-bottom:10px;">'
            '<span style="font-size:0.75rem; color:#64748b;'
            ' text-transform:uppercase; letter-spacing:0.06em;">Keywords detected</span><br>'
            '<div style="margin-top:6px;">'
            + kw_html +
            '</div></div>'

            # ── Sentiment score ───────────────────────────────────────────────
            '<div style="margin-bottom:10px;">'
            '<span style="font-size:0.75rem; color:#64748b;'
            ' text-transform:uppercase; letter-spacing:0.06em;">Sentiment score</span><br>'
            f'<span style="font-size:1.1rem; font-weight:700; color:{sent_color};">'
            f'{sentiment_score:+.2f}'
            '</span>'
            '<span style="font-size:0.78rem; color:#64748b;"> / 1.0 (VADER polarity)</span>'
            '</div>'

            # ── Interpretation ────────────────────────────────────────────────
            '<div>'
            '<span style="font-size:0.75rem; color:#64748b;'
            ' text-transform:uppercase; letter-spacing:0.06em;">Interpretation</span><br>'
            '<div style="margin-top:6px; font-size:0.84rem; color:#cbd5e1; line-height:1.7;">'
            + reason_html +
            '</div></div>'

            '</div>'   # close outer wrapper
        )

        st.markdown(explanation_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "chat_history" not in st.session_state:
    # List of {"role": "user"|"bot", "text": str}
    st.session_state.chat_history = []

if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None

if "last_explanation" not in st.session_state:
    st.session_state.last_explanation = None

if "emotion_memory" not in st.session_state:
    st.session_state.emotion_memory = InteractionMemory(max_entries=20)

if "last_trend" not in st.session_state:
    st.session_state.last_trend = None

if "training_logs" not in st.session_state:
    st.session_state.training_logs = []

if "is_training" not in st.session_state:
    st.session_state.is_training = False

if "training_complete" not in st.session_state:
    st.session_state.training_complete = model_is_trained()


# ---------------------------------------------------------------------------
# Simulated bot responses (rule-based, to create context for the predictor)
# ---------------------------------------------------------------------------

BOT_RESPONSES = {
    "Neutral":   [
        "That's good to hear! Is there anything specific on your mind?",
        "Sounds like things are pretty stable. How can I support you today?",
        "Thanks for sharing. Feel free to tell me more whenever you're ready.",
    ],
    "Stressed": [
        "It sounds like you're under a lot of pressure right now. Want to talk through it?",
        "That sounds really overwhelming. Is there anything specific I can help you break down?",
        "I can hear the stress in what you're sharing. Take a breath — you're not alone.",
    ],
    "Depressed": [
        "I'm really glad you felt comfortable sharing that. How long have you been feeling this way?",
        "That sounds incredibly heavy to carry. Please know that what you're feeling is valid.",
        "I hear you. Reaching out is a brave thing to do. Is there someone you trust you can talk to as well?",
    ],
}

import random as _random

def generate_bot_response(label: str) -> str:
    responses = BOT_RESPONSES.get(label, BOT_RESPONSES["Neutral"])
    return _random.choice(responses)


# ---------------------------------------------------------------------------
# Sidebar — Training panel
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🧠 Mental State Detector")
    st.markdown("---")

    # Model status indicator
    if model_is_trained():
        st.markdown("✅ **Model status:** Ready")
    else:
        st.markdown("⚠️ **Model status:** Not trained yet")
        st.markdown(
            '<div class="info-callout">'
            'No trained model found. Click <strong>Train Model Locally</strong> '
            'to generate synthetic data and fine-tune distilroberta-base.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### ⚙️ Training")

    epochs = st.slider("Training epochs", min_value=2, max_value=20, value=5)
    episodes = st.slider("Episodes per class", min_value=20, max_value=80, value=35)

    train_btn = st.button(
        "🚀 Train Model Locally",
        disabled=st.session_state.is_training,
        use_container_width=True,
    )

    # Training log area
    if st.session_state.training_logs:
        st.markdown("**Training log:**")
        log_text = "\n".join(st.session_state.training_logs[-30:])
        st.markdown(
            f'<div class="train-log">{log_text}</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.is_training:
        st.info("Training in progress … this may take several minutes on CPU.")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown(
        "**Model:** `distilroberta-base`  \n"
        "**Task:** 3-class sequence classification  \n"
        "**Context:** Last 3 conversation turns  \n"
        "**Labels:** Neutral · Stressed · Depressed  \n"
    )

    # Clear chat
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.last_prediction = None
        st.session_state.last_explanation = None
        st.session_state.emotion_memory.clear()
        st.session_state.last_trend = None
        st.rerun()


# ---------------------------------------------------------------------------
# Training trigger
# ---------------------------------------------------------------------------

if train_btn and not st.session_state.is_training:
    st.session_state.is_training = True
    st.session_state.training_logs = ["Starting training pipeline …"]
    st.session_state.training_complete = False
    st.rerun()

# Run the actual training (after rerun so UI shows spinner first)
if st.session_state.is_training and not st.session_state.training_complete:
    # Capture progress logs
    progress_placeholder = st.empty()

    with st.spinner("Training in progress — please wait …"):
        from src.train import train as run_training

        log_lines: list[str] = []

        def log_callback(msg: str):
            log_lines.append(msg)
            st.session_state.training_logs = log_lines.copy()

        try:
            run_training(
                episodes_per_class=episodes,
                epochs=epochs,
                progress_callback=log_callback,
            )
            st.session_state.training_complete = True
            st.session_state.training_logs.append("✓ Training complete!")
        except Exception as exc:
            st.session_state.training_logs.append(f"✗ Error: {exc}")

        st.session_state.is_training = False
        # Clear cached predictor so the new model is loaded
        load_predictor.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Main layout — two columns
# ---------------------------------------------------------------------------

col_chat, col_pred = st.columns([3, 2], gap="large")

# ---- Left column: chat interface ----
with col_chat:
    st.markdown("## 💬 Conversation")
    st.markdown(
        '<div class="info-callout">'
        'Type a message below. The AI will respond empathetically and the panel on '
        'the right will display your <strong>detected mental state</strong> in real time, '
        'based on your last 3 turns of context.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Render chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown(
                "<div style='color:#475569; text-align:center; padding:40px;'>"
                "Start the conversation by typing below …"
                "</div>",
                unsafe_allow_html=True,
            )
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="user-bubble">{msg["text"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="bot-bubble">🤖 {msg["text"]}</div>',
                    unsafe_allow_html=True,
                )

    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your message",
            placeholder="Tell me how you're feeling …",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send →", use_container_width=True)

    if submitted and user_input.strip():
        if not model_is_trained():
            st.error("Please train the model first using the sidebar button.")
        else:
            # Collect conversation context (only user turns, last CONTEXT_WINDOW)
            user_turns = [
                m["text"]
                for m in st.session_state.chat_history
                if m["role"] == "user"
            ]
            context = user_turns[-3:]  # last 3 user turns as context

            # Run prediction
            try:
                predictor = load_predictor()
                result = predictor.predict(
                    current_text=user_input.strip(),
                    context_history=context,
                )
                st.session_state.last_prediction = result
                # ── Explainability ────────────────────────────────────────
                st.session_state.last_explanation = _explain_prediction(
                    text=user_input.strip(),
                    prediction=result,
                )
                # ── Memory + Trend ────────────────────────────────────────
                st.session_state.emotion_memory.add(
                    text=user_input.strip(),
                    prediction=result,
                    explanation=st.session_state.last_explanation,
                )
                st.session_state.last_trend = _analyze_trend(
                    st.session_state.emotion_memory.entries
                )
            except Exception as exc:
                st.session_state.last_prediction = None
                st.session_state.last_explanation = None
                st.error(f"Prediction error: {exc}")
                result = None

            # Generate bot response
            bot_reply = generate_bot_response(
                result["label"] if result else "Neutral"
            )

            # Append to chat history
            st.session_state.chat_history.append(
                {"role": "user", "text": user_input.strip()}
            )
            st.session_state.chat_history.append(
                {"role": "bot", "text": bot_reply}
            )

            st.rerun()


# ---- Right column: prediction display ----
with col_pred:
    st.markdown("## 📊 Mental State Analysis")

    if st.session_state.last_prediction is None:
        st.markdown(
            "<div style='color:#475569; text-align:center; padding:60px 20px;'>"
            "<div style='font-size:3rem; margin-bottom:16px;'>🧠</div>"
            "<div>Predictions will appear here as you chat.</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        result = st.session_state.last_prediction
        render_prediction_card(result, st.session_state.last_explanation)

        # Context information
        user_turns = [
            m["text"]
            for m in st.session_state.chat_history
            if m["role"] == "user"
        ]
        if len(user_turns) > 1:
            context_shown = user_turns[-4:-1]  # up to 3 previous turns
            if context_shown:
                st.markdown("---")
                st.markdown(
                    "**Context used for prediction:**",
                )
                for i, ctx in enumerate(context_shown, 1):
                    st.markdown(
                        f'<div style="font-size:0.82rem; color:#64748b; '
                        f'padding:6px 10px; margin-bottom:4px; '
                        f'background:rgba(255,255,255,0.04); border-radius:6px;">'
                        f'[{i}] {ctx}</div>',
                        unsafe_allow_html=True,
                    )

        # Mental health disclaimer
        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.75rem; color:#475569; line-height:1.6;">'
            '⚠️ <strong>Disclaimer:</strong> This is a proof-of-concept ML system '
            'trained on synthetic data. It is not a clinical diagnostic tool and '
            'should not be used as a substitute for professional mental health support. '
            'If you are struggling, please reach out to a qualified professional.'
            '</div>',
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Emotional Trend section  (full-width, below the two columns)
# ---------------------------------------------------------------------------

memory = st.session_state.emotion_memory
total_entries = len(memory)

if total_entries > 0:
    st.markdown("---")
    st.markdown("## 📈 Emotional Trend")

    trend_data = st.session_state.last_trend

    if trend_data:
        trend      = trend_data["trend"]
        confidence = trend_data["confidence"]
        summary    = trend_data["summary"]
        stats      = trend_data.get("stats", {})

        # ── Trend badge ────────────────────────────────────────────────────
        _trend_style = {
            "improving": ("#10b981", "rgba(16,185,129,0.12)", "rgba(16,185,129,0.30)", "✅"),
            "worsening": ("#ef4444", "rgba(239,68,68,0.12)",  "rgba(239,68,68,0.30)",  "⚠️"),
            "stable":    ("#818cf8", "rgba(99,102,241,0.12)", "rgba(99,102,241,0.30)", "📊"),
        }
        t_color, t_bg, t_border, t_icon = _trend_style.get(
            trend, _trend_style["stable"]
        )

        # Top metric row
        tcol1, tcol2, tcol3 = st.columns([2, 1, 1])
        with tcol1:
            st.markdown(
                '<div style="background:' + t_bg + '; border:1px solid ' + t_border + ';'
                ' border-radius:14px; padding:14px 20px;">'
                '<div style="font-size:0.72rem; color:#64748b; text-transform:uppercase;'
                ' letter-spacing:0.08em; margin-bottom:6px;">Current Trend</div>'
                '<div style="font-size:1.4rem; font-weight:700; color:' + t_color + ';">'
                + t_icon + ' ' + trend.capitalize() +
                '</div></div>',
                unsafe_allow_html=True,
            )
        with tcol2:
            st.markdown(
                '<div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);'
                ' border-radius:14px; padding:14px 20px; text-align:center;">'
                '<div style="font-size:0.72rem; color:#64748b; text-transform:uppercase;'
                ' letter-spacing:0.08em; margin-bottom:6px;">Confidence</div>'
                '<div style="font-size:1.3rem; font-weight:700; color:#f0f4ff;">'
                + f'{confidence:.0%}' +
                '</div></div>',
                unsafe_allow_html=True,
            )
        with tcol3:
            st.markdown(
                '<div style="background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08);'
                ' border-radius:14px; padding:14px 20px; text-align:center;">'
                '<div style="font-size:0.72rem; color:#64748b; text-transform:uppercase;'
                ' letter-spacing:0.08em; margin-bottom:6px;">Messages</div>'
                '<div style="font-size:1.3rem; font-weight:700; color:#f0f4ff;">'
                + str(total_entries) +
                '</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Summary text ───────────────────────────────────────────────────
        import re as _re2
        summary_html = _re2.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", summary)
        summary_html = summary_html.replace("  \n", "<br>")
        st.markdown(
            '<div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);'
            ' border-radius:12px; padding:14px 20px; font-size:0.87rem; color:#cbd5e1; line-height:1.8;">'
            + summary_html +
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Sentiment chart ────────────────────────────────────────────────
        sentiment_series = memory.sentiment_series()
        distress_series  = memory.distress_series()

        if len(sentiment_series) >= 2:
            import plotly.graph_objects as go

            indices = list(range(1, len(sentiment_series) + 1))

            fig = go.Figure()

            # Sentiment line
            fig.add_trace(go.Scatter(
                x=indices,
                y=sentiment_series,
                mode="lines+markers",
                name="Sentiment",
                line=dict(color="#818cf8", width=2.5),
                marker=dict(size=7, color="#818cf8",
                            line=dict(color="#ffffff", width=1)),
                hovertemplate="Msg %{x}<br>Sentiment: %{y:.2f}<extra></extra>",
            ))

            # Zero reference line
            fig.add_hline(
                y=0, line_dash="dot",
                line_color="rgba(255,255,255,0.15)",
                annotation_text="neutral",
                annotation_font_color="#64748b",
                annotation_font_size=10,
            )

            # Colour bands
            fig.add_hrect(y0=-1.1, y1=0,  fillcolor="rgba(239,68,68,0.05)",  line_width=0)
            fig.add_hrect(y0=0,  y1=1.1,  fillcolor="rgba(16,185,129,0.05)", line_width=0)

            fig.update_layout(
                title=dict(
                    text="Sentiment Polarity Over Time",
                    font=dict(size=14, color="#94a3b8"),
                    x=0,
                ),
                xaxis=dict(
                    title="Message #",
                    color="#64748b",
                    gridcolor="rgba(255,255,255,0.06)",
                    tickmode="linear", dtick=1,
                    showline=False, zeroline=False,
                ),
                yaxis=dict(
                    title="Polarity",
                    color="#64748b",
                    range=[-1.1, 1.1],
                    gridcolor="rgba(255,255,255,0.06)",
                    showline=False, zeroline=False,
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                font=dict(family="Inter, sans-serif", color="#94a3b8"),
                legend=dict(
                    orientation="h", y=1.12,
                    font=dict(size=11, color="#94a3b8"),
                    bgcolor="rgba(0,0,0,0)",
                ),
                margin=dict(l=0, r=0, t=40, b=0),
                height=260,
            )

            st.plotly_chart(fig, use_container_width=True)

        # ── Label frequency mini-bar ───────────────────────────────────────
        if total_entries >= 2:
            lc = stats.get("label_counts", {})
            _lc_colors = {
                "Neutral":   ("#10b981", "rgba(16,185,129,0.18)"),
                "Stressed":  ("#f59e0b", "rgba(245,158,11,0.18)"),
                "Depressed": ("#ef4444", "rgba(239,68,68,0.18)"),
            }
            st.markdown(
                '<div style="display:flex; gap:10px; margin-top:4px;">'
                + "".join(
                    '<div style="flex:1; background:' + _lc_colors.get(lbl, ("#818cf8","rgba(99,102,241,0.18)"))[1] + ';'
                    ' border-radius:10px; padding:10px 14px; text-align:center;">'
                    '<div style="font-size:0.7rem; color:#64748b; text-transform:uppercase;'
                    ' letter-spacing:0.07em;">' + lbl + '</div>'
                    '<div style="font-size:1.4rem; font-weight:700; color:' + _lc_colors.get(lbl, ("#818cf8",""))[0] + ';">' + str(cnt) + '</div>'
                    '</div>'
                    for lbl, cnt in lc.items()
                )
                + '</div>',
                unsafe_allow_html=True,
            )
