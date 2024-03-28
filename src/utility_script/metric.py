# https://www.kaggle.com/code/conjuring92/pii-metric-fine-grained-eval

from typing import Dict
from src.utils import parse_predictions
import pandas as pd


class PRFScore:
    """A precision / recall / F score."""

    def __init__(
        self,
        *,
        tp: int = 0,
        fp: int = 0,
        fn: int = 0,
    ) -> None:
        self.tp = tp
        self.fp = fp
        self.fn = fn

    def __len__(self) -> int:
        return self.tp + self.fp + self.fn

    def __iadd__(self, other):  # in-place add
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        return self

    def __add__(self, other):
        return PRFScore(
            tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn
        )

    def score_set(self, cand: set, gold: set) -> None:
        self.tp += len(cand.intersection(gold))
        self.fp += len(cand - gold)
        self.fn += len(gold - cand)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp + 1e-100)

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn + 1e-100)

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * ((p * r) / (p + r + 1e-100))

    @property
    def f5(self) -> float:
        beta = 5
        p = self.precision
        r = self.recall

        fbeta = (1 + (beta**2)) * p * r / ((beta**2) * p + r + 1e-100)
        return fbeta

    def to_dict(self) -> Dict[str, float]:
        return {"p": self.precision, "r": self.recall, "f5": self.f5}


def get_triplets(df: pd.DataFrame):
    return {(row.document, row.token, row.label) for row in df.itertuples()}


def compute_metrics(p, id2label, valid_ds, valid_df, threshold=0.9, softmax=True):
    """
    Compute the LB metric (lb) and other auxiliary metrics
    """
    predictions, _ = p

    pred_df = parse_predictions(predictions, id2label, valid_ds, threshold=threshold, softmax=softmax)

    references, predictions = set(get_triplets(valid_df)), get_triplets(pred_df)

    score_per_type = initialize_score_per_type()

    count_tp_and_fp(predictions, references, score_per_type)
    count_fn(references, score_per_type)

    totals = calculate_total_metrics(score_per_type)
    final_results = parse_metric(score_per_type, totals)

    return final_results


def initialize_score_per_type():
    score_per_type = {}
    types = [
        'B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 
        'B-STREET_ADDRESS', 'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM', 
        'I-NAME_STUDENT', 'I-PHONE_NUM', 'I-STREET_ADDRESS', 'I-URL_PERSONAL'
    ]

    for value in types:
        if value != "O":
            value = value[2:]  # avoid B- and I- prefix

        score_per_type[value] = PRFScore()

    return score_per_type


def parse_metric(score_per_type, totals):
    results = {
        "ents_p": totals.precision,
        "ents_r": totals.recall,
        "ents_f5": totals.f5,
        "ents_per_type": {
            k: v.to_dict() for k, v in score_per_type.items() if k != "O"
        },
    }

    # Unpack nested dictionaries
    final_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for n, v in value.items():
                if isinstance(v, dict):
                    for n2, v2 in v.items():
                        final_results[f"{key}_{n}_{n2}"] = v2
                else:
                    final_results[f"{key}_{n}"] = v
        else:
            final_results[key] = value

    return final_results


def calculate_total_metrics(score_per_type):
    totals = PRFScore()

    for prf in score_per_type.values():
        totals += prf

    return totals


def count_tp_and_fp(predictions, references, score_per_type):
    for ex in predictions:
        pred_type = ex[-1]  # (document, token, label)

        if pred_type != "O":
            pred_type = pred_type[2:]  # avoid B- and I- prefix

        if ex in references:
            score_per_type[pred_type].tp += 1
            references.remove(ex)
        else:
            score_per_type[pred_type].fp += 1


def count_fn(references, score_per_type):
    for _, _, ref_type in references:
        if ref_type != "O":
            ref_type = ref_type[2:]  # avoid B- and I- prefix

        score_per_type[ref_type].fn += 1