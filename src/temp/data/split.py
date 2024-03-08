import pandas as pd
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed


from src.utilities.text import rebuild_text


def stride_split_rows(
    df: pd.DataFrame,
    max_length: int,
    stride_length: int,
) -> pd.DataFrame:
    new_df = []

    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(_process_row, max_length, stride_length, row)
            for _, row in df.iterrows()
        ]

        for future in as_completed(futures):
            for output in future.result():
                new_df.append(output)

    return pd.DataFrame(new_df)


def _process_row(max_length, stride_length, row):
    output = []
    tokens = row["tokens"]

    if len(tokens) > max_length:
        # Split row only when it has more than max_lengh tokens
        start = 0

        while start < len(tokens):
            remaining_tokens = len(tokens) - start

            if remaining_tokens < max_length and start != 0:
                # Adjust start for the last window to ensure it has max_length tokens
                start = max(0, len(tokens) - max_length)

            end = min(start + max_length, len(tokens))
            output.append(_create_new_row(row, start, end))

            if remaining_tokens >= max_length:
                start += stride_length
            else:
                # Break the loop if we've adjusted for the last window
                break
    else:
        output.append(_create_new_row(row))

    return output


def _create_new_row(
    row: pd.Series,
    start: int = None,
    end: int = None,
):
    tokens = row["tokens"]
    trailing_whitespace = row["trailing_whitespace"]
    labels = row["labels"]
    token_indices = row["token_indices"]
    full_text = row["full_text"]

    if start is not None and end is not None:
        tokens = tokens[start:end]
        trailing_whitespace = trailing_whitespace[start:end]
        labels = labels[start:end]
        token_indices = list(range(start, end))
        full_text = rebuild_text(tokens, trailing_whitespace)

    return {
        "document": row["document"],
        "valid": row["valid"],
        "source": row["source"],
        "tokens": tokens,
        "trailing_whitespace": trailing_whitespace,
        "labels": labels,
        "token_indices": token_indices,
        "full_text": full_text,
    }


# if __name__ == "__main__":
#     from src.data import split_rows as original_split

#     test_df = pd.read_csv("Dataset/competition/test.json")
#     max_length = 1024
#     stride_length = 386

#     actual = original_split(test_df, max_length, stride_length)
#     new = stride_split_rows(test_df, max_length, stride_length)

#     assert actual.shape == new.shape, "Shape doesn't match"

#     for r1, r2 in zip(actual.iterrows(), new.iterrows()):
#         r1, r2 = r1[1], r2[1]

#         assert r1["document"] == r2["document"], "Document not the same"
#         assert r1["valid"] == r2["valid"], "Valid not the same"
#         assert r1["source"] == r2["source"], "Source not the same"
#         assert r1["tokens"] == r2["tokens"], "Tokens not the same"
#         assert (
#             r1["trailing_whitespace"] == r2["trailing_whitespace"]
#         ), "Trailing Whitespace not the same"
#         assert r1["labels"] == r2["labels"], "Labels not the same"
#         assert r1["token_indices"] == r2["token_indices"], "Token indices not the same"
#         assert r1["full_text"] == r2["full_text"], "Full text not the same"

#     print("All tests passed!")
