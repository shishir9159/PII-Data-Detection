import pandas as pd
from typing import List

_RANDOM_STATE = 29

_PII_TYPES = [
    "EMAIL",
    "ID_NUM",
    "NAME_STUDENT",
    "PHONE_NUM",
    "STREET_ADDRESS",
    "URL_PERSONAL",
    "USERNAME",
]


def oversample(df: pd.DataFrame):
    pass


def stratified_split(df: pd.DataFrame, folds: int) -> pd.DataFrame:
    df["fold_no"] = 0
    df["taken"] = 0
    df["unique_label_count"] = df.unique_labels.map(len)

    df = stratified_sample_positives(df, folds)
    df = stratified_sample_negatives(df, folds)

    df.drop(columns=["taken", "unique_label_count"], inplace=True)

    return df


def stratified_sample_positives(df: pd.DataFrame, folds: int) -> pd.DataFrame:
    for fold in range(folds):
        for source in df.source.unique():
            for pii_type in _PII_TYPES:
                mask = (
                    (df.source == source)
                    & (df[pii_type] == 1)
                    & (df.taken < df.unique_label_count)
                )
                rows = df[mask].sample(
                    frac=1 / folds,
                    random_state=_RANDOM_STATE,
                    weights=_calculate_sampling_weight(df, mask),
                )
                df.loc[rows.index, "taken"] = df.loc[rows.index, "taken"] + 1
                df.loc[rows.index, "fold"] = fold

    # df.loc[(df.NO_PII == 0) & (df.taken < df.unique_label_count), "fold"] = fold
    return df


def stratified_sample_negatives(df: pd.DataFrame, folds: int) -> List[pd.DataFrame]:
    for fold in range(folds):
        for source in df.source.unique():
            mask = (df.NO_PII == 1) & (df.source == source) & (df.taken == 0)
            rows = df[mask].sample(frac=1 / folds, random_state=_RANDOM_STATE)
            df.loc[rows.index, "taken"] = 1
            df.loc[rows.index, "fold"] = fold

    # df.loc[(df.NO_PII == 1) & (df.taken == 0), "fold"] = fold
    return df


def _calculate_sampling_weight(df, mask):
    weights = df[mask].taken
    weights = 1 / (weights + 1e-6) if any(weights > 0) else None
    return weights
