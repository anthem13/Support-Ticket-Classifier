import random

from tickets.model import TicketClassifier
from tickets.perturb import PERTURBATIONS
from tickets.synthetic import generate


def test_trains_and_predicts():
    clf = TicketClassifier().fit(generate(600))
    out = clf.predict(["I was charged twice for my Pro subscription", ""])
    assert out[0]["category"] == "billing"
    assert out[1]["needs_human"] is True


def test_perturbations_return_strings():
    rng = random.Random(0)
    for fn in PERTURBATIONS.values():
        assert isinstance(fn("my order has not arrived", rng), str)
