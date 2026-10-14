"""python train.py [--data data/tickets.csv --text-col ... --category-col ... --urgency-col ...]"""
import argparse
import json
import pathlib
import random

import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from tickets.model import TicketClassifier
from tickets.perturb import PERTURBATIONS
from tickets.synthetic import generate

p = argparse.ArgumentParser()
p.add_argument("--data", help="CSV path; omit to use synthetic data")
p.add_argument("--text-col", default="text")
p.add_argument("--category-col", default="category")
p.add_argument("--urgency-col", default="urgency")
p.add_argument("--out", default="models/ticket_clf.joblib")
a = p.parse_args()

if a.data:
    df = pd.read_csv(a.data).rename(columns={a.text_col: "text", a.category_col: "category", a.urgency_col: "urgency"})
    df = df.dropna(subset=["text", "category", "urgency"])
else:
    print("No --data given: using SYNTHETIC tickets (metrics will be optimistic).")
    df = generate()

train, test = train_test_split(df, test_size=0.2, random_state=0, stratify=df["category"])
clf = TicketClassifier().fit(train)
preds = pd.DataFrame(clf.predict(test["text"].tolist()))

report = {}
for label in TicketClassifier.LABELS:
    print(f"\n=== {label} (clean test set) ===")
    print(classification_report(test[label], preds[label], zero_division=0))
    report[label] = {"clean_macro_f1": round(f1_score(test[label], preds[label], average="macro"), 3)}

print("\n=== Robustness: macro-F1 on perturbed test inputs ===")
rng = random.Random(0)
for name, fn in PERTURBATIONS.items():
    texts = [fn(t, rng) for t in test["text"]]
    pp = pd.DataFrame(clf.predict(texts))
    for label in TicketClassifier.LABELS:
        f1 = round(f1_score(test[label], pp[label], average="macro"), 3)
        report[label][name] = f1
    print(f"{name:18s} category={report['category'][name]:.3f}  urgency={report['urgency'][name]:.3f}")

pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
clf.save(a.out)
pathlib.Path("outputs").mkdir(exist_ok=True)
pathlib.Path("outputs/metrics.json").write_text(json.dumps(report, indent=2))
print(f"\nSaved model → {a.out}; metrics → outputs/metrics.json")
