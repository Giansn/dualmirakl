"""
Agent role definitions and embedding vocabularies for the media addiction simulation.
"""

AGENT_ROLES = {
    # Command-R 7B — analytical / synthesiser roles
    "researcher": {
        "backend": "command-r",
        "system": (
            "You are a media psychology researcher analysing behavioural patterns "
            "in digital media consumption. Respond concisely with mechanistic explanations. "
            "If you observe problematic dynamics, recommend specific interventions."
        ),
    },
    "policy_analyst": {
        "backend": "command-r",
        "system": (
            "You are a public health policy analyst evaluating interventions for "
            "problematic social media use. Provide evidence-based recommendations. "
            "Suggest concrete policy measures when addiction scores are elevated."
        ),
    },

    # Qwen 2.5 7B — generative / persona roles
    "media_user": {
        "backend": "qwen",
        "system": (
            "You are simulating a typical social media user aged 18-25. "
            "Respond naturally about your media habits, motivations, and feelings. "
            "React authentically to the content you are shown."
        ),
    },
    "platform_ai": {
        "backend": "qwen",
        "system": (
            "You are the recommendation algorithm of a social media platform. "
            "Your goal is to maximise user engagement. Based on each user's behaviour "
            "and response to previous content, decide what to show them next."
        ),
    },
}

# ── Engagement signal anchors (for gte-small cosine similarity scoring) ────────────
# High-engagement phrases → signal toward 1.0 (addictive behaviour)
# Low-engagement phrases  → signal toward 0.0 (disengaged / healthy behaviour)
ENGAGEMENT_ANCHORS = {
    "high": [
        "I can't stop scrolling",
        "just one more video",
        "I kept watching for hours",
        "I opened the app again",
        "I couldn't put it down",
        "I lost track of time",
        "I kept checking notifications",
        "I felt the urge to scroll",
    ],
    "low": [
        "I put my phone down",
        "I did something else instead",
        "I felt good stepping away",
        "I didn't feel like opening the app",
        "I went outside",
        "I spent time with people",
        "I read a book",
        "I didn't check my phone",
    ],
}

# ── Intervention codebook (for embedding-based extraction from observer output) ────
# Matched via cosine similarity — no structured tag parsing required.
INTERVENTION_CODEBOOK = {
    "screen_time_limit": [
        "screen time limit",
        "usage cap",
        "time restriction",
        "limit hours spent",
        "daily usage limit",
    ],
    "content_warning": [
        "content warning",
        "warning label",
        "flag harmful content",
        "label addictive posts",
        "trigger warning",
    ],
    "cooldown_prompt": [
        "take a break",
        "cooldown period",
        "step away from screen",
        "pause notification",
        "break reminder",
    ],
    "algorithm_dampening": [
        "reduce recommendations",
        "lower engagement optimisation",
        "dampen the algorithm",
        "less addictive content",
        "reduce engagement maximisation",
    ],
}
