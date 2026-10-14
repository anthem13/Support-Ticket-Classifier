"""TF-IDF (word + char n-grams) → logistic regression, one head per label."""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


def make_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
        # char n-grams make the model far more robust to typos and odd tokens
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])
    return Pipeline([("features", features),
                     ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0))])


class TicketClassifier:
    LABELS = ("category", "urgency")

    def __init__(self):
        self.heads = {label: make_pipeline() for label in self.LABELS}

    def fit(self, df: pd.DataFrame) -> "TicketClassifier":
        for label, pipe in self.heads.items():
            pipe.fit(df["text"], df[label])
        return self

    def predict(self, texts: list[str]) -> list[dict]:
        out = [{} for _ in texts]
        for label, pipe in self.heads.items():
            probs = pipe.predict_proba(texts)
            for row, p in zip(out, probs):
                i = p.argmax()
                row[label] = pipe.classes_[i]
                row[f"{label}_confidence"] = round(float(p[i]), 3)
        for row, text in zip(out, texts):
            # Low-confidence or empty tickets go to a human instead of being silently misrouted.
            row["needs_human"] = (not text.strip()) or row["category_confidence"] < 0.5
        return out

    def save(self, path: str) -> None:
        joblib.dump(self.heads, path)

    @classmethod
    def load(cls, path: str) -> "TicketClassifier":
        obj = cls()
        obj.heads = joblib.load(path)
        return obj
