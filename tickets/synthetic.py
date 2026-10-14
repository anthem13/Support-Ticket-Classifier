"""Synthetic labelled tickets so the pipeline runs end-to-end before you download a real dataset.

Replace with a real dataset (see README) for the portfolio version — synthetic data makes
metrics look better than they are.
"""
import random

import pandas as pd

TEMPLATES = {
    "billing": [
        "I was charged twice for my {plan} subscription this month",
        "refund request for invoice {num}, the amount is wrong",
        "why did my card get billed {amt} when I cancelled",
        "need a copy of my invoice for {month}",
        "payment failed but money was deducted from my account",
    ],
    "technical": [
        "the app crashes every time I open the {feature} page",
        "getting error {num} when uploading files",
        "{feature} is not loading, just a blank screen",
        "sync stopped working after the latest update",
        "API returns 500 on the {feature} endpoint",
    ],
    "account": [
        "can't log in, password reset email never arrives",
        "please delete my account and all my data",
        "how do I change the email on my account",
        "my account got locked after too many attempts",
        "need to add a teammate to our {plan} workspace",
    ],
    "shipping": [
        "my order {num} hasn't arrived yet, it's been {days} days",
        "package shows delivered but I never received it",
        "can I change the delivery address for order {num}",
        "item arrived damaged, box was crushed",
        "tracking number for order {num} doesn't work",
    ],
    "feedback": [
        "love the new {feature} feature, great job",
        "would be nice if you supported dark mode",
        "suggestion: let us export reports to excel",
        "the new design is confusing, please bring back the old {feature}",
        "just wanted to say your support team was really helpful",
    ],
}
URGENT = ["URGENT", "asap", "production is down", "we are losing money", "immediately", "critical"]
CALM = ["no rush", "when you get a chance", "just curious", "fyi", ""]
FILL = {
    "plan": ["Pro", "Team", "Enterprise", "Basic"], "num": ["4821", "ORD-99812", "E1043", "#5520"],
    "amt": ["$49", "$120", "₹999"], "month": ["March", "July", "last month"],
    "feature": ["dashboard", "reports", "billing", "export", "search"], "days": ["5", "10", "14"],
}


def generate(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        category = rng.choice(list(TEMPLATES))
        text = rng.choice(TEMPLATES[category]).format(**{k: rng.choice(v) for k, v in FILL.items()})
        # Urgency: driven by cue words, with label noise so it's not trivially separable.
        r = rng.random()
        if r < 0.25:
            urgency, text = "high", f"{rng.choice(URGENT)} {text}"
        elif r < 0.6:
            urgency = "medium"
        else:
            urgency, text = "low", f"{text} {rng.choice(CALM)}".strip()
        if category in ("feedback",) and urgency == "high" and rng.random() < 0.7:
            urgency = "low"
        if rng.random() < 0.05:  # label noise
            urgency = rng.choice(["low", "medium", "high"])
        rows.append({"text": text, "category": category, "urgency": urgency})
    return pd.DataFrame(rows)
