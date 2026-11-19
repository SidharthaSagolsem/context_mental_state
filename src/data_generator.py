"""
src/data_generator.py
=====================
Generates a realistic synthetic dataset of multi-turn conversational episodes
labeled with mental states: Neutral (0), Stressed (1), Depressed (2).

Each "episode" simulates a multi-turn conversation between two speakers (User and Bot).
Labels are applied to the USER's turns based on the episode's mental-state theme.

Output: pandas DataFrame with columns:
    - episode_id  : int  — unique identifier for each conversation episode
    - turn_id     : int  — sequential index of the turn within the episode
    - speaker     : str  — "User" or "Bot"
    - text        : str  — utterance text
    - label       : int  — mental state label (only meaningful for "User" turns;
                           Bot turns receive label -1 as a sentinel)
"""

import random
import pandas as pd

# ---------------------------------------------------------------------------
# Seed for reproducibility
# ---------------------------------------------------------------------------
random.seed(42)

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------
NEUTRAL = 0
STRESSED = 1
DEPRESSED = 2

LABEL_NAMES = {NEUTRAL: "Neutral", STRESSED: "Stressed", DEPRESSED: "Depressed"}

# ---------------------------------------------------------------------------
# Utterance banks — varied enough to prevent the model from simply memorising
# a handful of phrases.  Each bank holds (user_utterance, bot_response) pairs.
# ---------------------------------------------------------------------------

NEUTRAL_PAIRS = [
    ("Hey, how's your day going?",                          "Pretty good! Just catching up on a few things."),
    ("I finished reading that book you recommended.",       "Oh nice! What did you think of it?"),
    ("The weather outside is actually quite pleasant today.","Sounds lovely — maybe a good day for a walk?"),
    ("I just had lunch. Nothing fancy, but it was good.",   "Sometimes simple is best! What did you have?"),
    ("I've been learning a bit of guitar lately.",          "That's cool! How long have you been at it?"),
    ("Any good TV shows you'd recommend?",                  "I've heard good things about several recent ones."),
    ("I organised my desk and it already feels better.",    "A tidy space really does help with focus."),
    ("Had a nice chat with my neighbour this morning.",     "That's always a pleasant way to start the day."),
    ("I think I'll cook something new tonight.",            "Experimenting in the kitchen can be fun!"),
    ("Just got back from a short walk in the park.",        "Fresh air can really clear your head."),
    ("Nothing much going on today.",                        "Sometimes a quiet day is exactly what you need."),
    ("I set a small goal for myself this week.",            "Small goals are a great way to stay motivated."),
    ("Caught up with an old friend online earlier.",        "That must have been nice — it's good to reconnect."),
    ("The morning coffee was just right today.",            "A good coffee really sets the tone, doesn't it?"),
    ("I finished a puzzle I've been working on.",           "Very satisfying! Any new ones lined up?"),
    ("Work was steady, nothing too eventful.",              "A calm workday can feel like a gift sometimes."),
    ("I'm thinking about starting a journal.",              "Writing can be a great way to process things."),
    ("Just browsing the internet, nothing in particular.",  "Sometimes that's exactly the right pace."),
    ("I tried meditating for ten minutes this morning.",    "Even a few minutes can make a real difference."),
    ("Planning a small get-together with friends.",         "That sounds like something to look forward to!"),
]

STRESSED_PAIRS = [
    ("I have three deadlines tomorrow and I haven't started.", "That does sound overwhelming. Is there one you can tackle first?"),
    ("My boss keeps changing the requirements at the last minute.", "That kind of unpredictability is incredibly frustrating."),
    ("I can't sleep — I keep running through my to-do list.", "Sleep deprivation on top of stress is really hard. Have you tried a short wind-down routine?"),
    ("I snapped at my partner this morning. I hate that.",  "It happens when we're stretched thin. Acknowledging it is already a step."),
    ("Everything is piling up and I don't know where to start.", "Breaking things into tiny pieces can help. What's the single most urgent task?"),
    ("I feel like I'm behind on everything at once.",       "That 'behind on everything' feeling is exhausting. What's one thing you can cross off today?"),
    ("The project is going to fail because of someone else's mistake.", "That helplessness when it's out of your control is really stressful."),
    ("I've been skipping meals because I don't have time.", "Please try to eat something — even a small snack. Your brain needs fuel."),
    ("My inbox has 200 unread emails and it's growing.",   "Email avalanche is real. Could you carve out 20 minutes to triage it?"),
    ("I haven't had a day off in three weeks.",             "That kind of sustained output without rest is unsustainable. Is there any gap soon?"),
    ("I'm worried I'm going to get blamed for the delay.",  "That anticipatory dread is exhausting. Have you been able to document your progress?"),
    ("My heart races every time my phone buzzes.",          "That hyper-vigilance is your nervous system on high alert. Have you been able to breathe?"),
    ("I'm supposed to present tomorrow and I'm panicking.", "Presentation anxiety is really common. Would running through it once help?"),
    ("I think I over-committed and now I'm drowning.",      "Recognising that is important. Could you renegotiate any of those commitments?"),
    ("I feel like everyone else is coping fine but me.",    "That perception is rarely accurate — many people hide their stress well."),
    ("I keep making small mistakes because I'm so frazzled.", "Fatigue really does hurt accuracy. Mandatory micro-breaks could help."),
    ("There's just too much to juggle right now.",          "Juggling too many things at once is a recipe for stress. What could you hand off or pause?"),
    ("I had a panic attack in the bathroom at work today.", "I'm really sorry to hear that. Are you safe right now? Talking to someone you trust can help."),
    ("My team is relying on me and I'm running on empty.",  "It's hard to give from an empty cup. Please try to rest tonight."),
    ("I can't focus on anything — my mind is everywhere.",  "Scattered attention under stress is normal. A quick five-minute walk can reset you."),
]

DEPRESSED_PAIRS = [
    ("I don't really see the point in getting out of bed anymore.", "I hear you. That kind of heaviness is really difficult. Would you like to talk about it?"),
    ("Everything feels grey lately. Nothing interests me.",  "That emotional numbness is really painful. How long have you been feeling this way?"),
    ("I cancelled on my friends again. I just couldn't face it.", "Isolation can make things feel heavier. It's okay — is there one small step you could take?"),
    ("I've been thinking a lot about how much of a burden I am.", "You are not a burden. Those thoughts can be really convincing, but they aren't facts."),
    ("I barely ate today. Food just doesn't appeal to me.",  "Appetite changes are really common with low mood. Can you manage something small?"),
    ("I used to love painting but I haven't touched it in months.", "Losing interest in things you loved is really hard. No pressure — it can return."),
    ("I wake up already exhausted. There's no reason to feel this tired.", "That kind of fatigue that sleep doesn't fix is worth paying attention to."),
    ("I keep replaying old mistakes. I can't let them go.", "Rumination is cruel and tiring. Those events are in the past, even if your mind keeps visiting them."),
    ("I feel completely disconnected from the people around me.", "That sense of disconnection is really lonely. Have you been able to share this with anyone?"),
    ("Some days I feel like things will never get better.",  "I'm glad you said that out loud. It's important to have support when you're feeling that way."),
    ("I don't enjoy anything anymore, even things that used to excite me.", "Anhedonia — losing the ability to feel pleasure — is one of the harder symptoms."),
    ("I've been crying and I'm not even sure why.",         "Sometimes sadness doesn't need a clear reason. It's still real and valid."),
    ("I feel invisible. Like I could disappear and nobody would notice.", "That feeling of invisibility is painful. I notice you, and I'm glad you're talking."),
    ("Getting through a single day feels like an enormous effort.", "When each day feels like climbing a mountain, even showing up takes real courage."),
    ("I haven't responded to messages in days. I don't have the energy.", "Low energy for social interaction is very common with depression. No judgement here."),
    ("Everything I try seems to fail. Maybe I'm just not good enough.", "Depression distorts our view of ourselves. Those thoughts are symptoms, not facts."),
    ("I've lost a lot of weight without trying. I just forget to eat.", "Significant unintended weight loss is something worth mentioning to a doctor."),
    ("I feel like a hollow shell of who I used to be.",     "That loss of self is one of the most painful parts. But the person you were is still there."),
    ("I don't see a future for myself.",                    "When the future looks blank, it can be frightening. Please consider reaching out to a counsellor."),
    ("I'm here but I'm not really here, if that makes sense.", "That detachment makes complete sense. You deserve real support right now."),
]


def _build_episode(episode_id: int, pairs: list, label: int, num_turns: int = 4) -> list[dict]:
    """
    Build a single conversational episode from a list of (user, bot) pairs.

    Args:
        episode_id: Unique identifier for this episode.
        pairs:      List of (user_utterance, bot_response) tuples.
        label:      Mental state label for user turns in this episode.
        num_turns:  Number of (user, bot) exchange rounds to include.

    Returns:
        List of row dicts ready to be appended to the main record list.
    """
    rows = []
    selected = random.sample(pairs, min(num_turns, len(pairs)))
    turn_id = 0
    for user_text, bot_text in selected:
        rows.append({
            "episode_id": episode_id,
            "turn_id": turn_id,
            "speaker": "User",
            "text": user_text,
            "label": label,
        })
        turn_id += 1
        rows.append({
            "episode_id": episode_id,
            "turn_id": turn_id,
            "speaker": "Bot",
            "text": bot_text,
            "label": -1,  # Bot turns are not classified
        })
        turn_id += 1
    return rows


def generate_dataset(
    episodes_per_class: int = 40,
    turns_per_episode: int = 4,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Generate a synthetic multi-turn conversational dataset.

    Args:
        episodes_per_class: Number of conversation episodes to generate per
                            mental-state class (Neutral / Stressed / Depressed).
                            Total episodes = 3 × episodes_per_class.
        turns_per_episode:  Number of (user, bot) exchange rounds per episode.
        output_path:        If provided, saves the DataFrame as a CSV at this path.

    Returns:
        pandas DataFrame with columns:
            episode_id, turn_id, speaker, text, label
    """
    all_rows: list[dict] = []
    episode_id = 0

    # Map each class to its utterance bank
    class_config = [
        (NEUTRAL,  NEUTRAL_PAIRS),
        (STRESSED, STRESSED_PAIRS),
        (DEPRESSED, DEPRESSED_PAIRS),
    ]

    for label, pairs in class_config:
        for _ in range(episodes_per_class):
            rows = _build_episode(
                episode_id=episode_id,
                pairs=pairs,
                label=label,
                num_turns=turns_per_episode,
            )
            all_rows.extend(rows)
            episode_id += 1

    df = pd.DataFrame(all_rows)

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"[data_generator] Dataset saved to {output_path}  "
              f"({len(df)} rows, {episode_id} episodes)")

    return df


# ---------------------------------------------------------------------------
# Quick sanity-check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = generate_dataset(output_path="synthetic_dataset.csv")
    print(df.head(12))
    print("\nLabel distribution (User turns only):")
    user_df = df[df["speaker"] == "User"]
    print(user_df["label"].value_counts().sort_index().rename(LABEL_NAMES))
