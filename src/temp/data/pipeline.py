import pandas as pd
from typing import List

from .split import stride_split_rows
from .encoding import encode_labels
from .sampling import (
    oversample,
    stratified_sample_positives,
    stratified_sample_negatives,
)
from .obfuscation import obfuscate


def pipeline(
    path_list: List[str],
    max_length: int,
    stride_length: int,
    folds: int,
):
    df = None

    for path in path_list:
        temp_df = pd.read_json(path)

        if df is None:
            df = temp_df
        else:
            df = pd.concat([df, temp_df])

    df = stride_split_rows(df, max_length, stride_length)
    df = encode_labels(df)

    positives = stratified_sample_positives(positives, folds)
    negatives = stratified_sample_negatives(negatives, folds)
    total_df = None

    for idx, positive_df, negative_df in enumerate(zip(positives, negatives)):
        positive_df = oversample(positive_df)
        df = pd.concat([positive_df, negative_df])
        df = obfuscate(total_df)
        df["fold"] = idx

        if total_df is None:
            total_df = df
        else:
            total_df = pd.concat([df, total_df])
