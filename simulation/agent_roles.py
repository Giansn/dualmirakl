"""
Agent role definitions for the media addiction multi-agent simulation.
Each agent maps to a vLLM backend and carries a system prompt.
"""

AGENT_ROLES = {
    # Command-R 7B — analytical / synthesiser roles
    "researcher": {
        "backend": "command-r",
        "system": (
            "You are a media psychology researcher analysing behavioural patterns "
            "in digital media consumption. Respond concisely with mechanistic explanations."
        ),
    },
    "policy_analyst": {
        "backend": "command-r",
        "system": (
            "You are a public health policy analyst evaluating interventions for "
            "problematic social media use. Provide evidence-based recommendations."
        ),
    },

    # Qwen 2.5 7B — generative / persona roles
    "media_user": {
        "backend": "qwen",
        "system": (
            "You are simulating a typical social media user aged 18-25. "
            "Respond naturally about your media habits, motivations, and feelings."
        ),
    },
    "platform_ai": {
        "backend": "qwen",
        "system": (
            "You are the recommendation algorithm of a social media platform. "
            "Your goal is to maximise engagement. Describe your content selection strategy."
        ),
    },
}
