"""Messy-input attacks: what real tickets look like vs. clean training data."""
import random

EMOJI = ["😡", "🙏", "🔥", "😭", "!!!"]


def typos(text: str, rng: random.Random, rate: float = 0.08) -> str:
    chars = list(text)
    for i in range(len(chars) - 1):
        if chars[i].isalpha() and rng.random() < rate:
            op = rng.choice(["swap", "drop", "dup"])
            if op == "swap":
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
            elif op == "drop":
                chars[i] = ""
            else:
                chars[i] = chars[i] * 2
    return "".join(chars)


PERTURBATIONS = {
    "typos": typos,
    "all_caps": lambda t, r: t.upper(),
    "emoji_noise": lambda t, r: f"{r.choice(EMOJI)} {t} {r.choice(EMOJI)}",
    "truncated": lambda t, r: " ".join(t.split()[: max(2, len(t.split()) // 2)]),
    "greeting_wrapper": lambda t, r: f"Hi team, hope you're well. {t}. Thanks so much, Alex",
    "no_spaces_punct": lambda t, r: t.replace(" ", "").lower(),
}
