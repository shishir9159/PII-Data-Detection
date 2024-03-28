import numpy as np
from datasets import Dataset
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def encode_labels(df):
    df["unique_labels"] = df["labels"].apply(
        lambda x: list(set([l.split("-")[1] for l in x if l != "O"]))
    )

    # add 1-hot encoding
    mlb = MultiLabelBinarizer()
    one_hot_encoded = mlb.fit_transform(df["unique_labels"])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=mlb.classes_)
    df = pd.concat([df, one_hot_df], axis=1)

    # add 'OTHER' column
    df["NO_PII"] = df["unique_labels"].apply(lambda x: 1 if len(x) == 0 else 0)

    return df, list(mlb.classes_) + ["NO_PII"]


def rebuild_text(tokens, trailing_whitespace):
    text = ""

    for token, ws in zip(tokens, trailing_whitespace):
        ws = " " if ws == True else ""
        text += token + ws

    return text


def rebuild_from_example(example):
    text, labels, token_map = [], [], []

    for idx, (t, l, ws) in enumerate(
        zip(example["tokens"], example["labels"], example["trailing_whitespace"])
    ):
        text.append(t)
        token_map.extend([idx] * len(t))
        labels.extend([l] * len(t))

        if ws:
            text.append(" ")
            labels.append("O")
            token_map.append(-1)

    labels = np.array(labels)
    text = "".join(text)

    return text, labels, token_map


def tokenize(example, tokenizer, label2id, max_length):
    text, labels, token_map = rebuild_from_example(example)

    # actual tokenization
    tokenized = tokenizer(
        text, return_offsets_mapping=True, max_length=max_length, truncation=True
    )

    token_labels = []

    for start_idx, end_idx in tokenized.offset_mapping:
        # CLS token
        if start_idx == 0 and end_idx == 0:
            token_labels.append(label2id["O"])
            continue

        # case when token starts with whitespace
        if text[start_idx].isspace():
            start_idx += 1

        token_labels.append(label2id[labels[start_idx]])

    length = len(tokenized.input_ids)

    return {
        **tokenized,
        "labels": token_labels,
        "length": length,
        "token_map": token_map,
    }


def create_dataset(data, tokenizer, max_length, label2id):
    columns = [
        "full_text",
        "document",
        "tokens",
        "trailing_whitespace",
        "labels",
        "token_indices",
    ]
    ds = Dataset.from_pandas(data[columns])
    ds = ds.map(
        tokenize,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": label2id,
            "max_length": max_length,
        },
        num_proc=6,
    )
    return ds


def split_rows(df, max_length, doc_stride):
    new_df = []

    for _, row in df.iterrows():
        tokens = row["tokens"]

        if len(tokens) > max_length:
            start = 0

            while start < len(tokens):
                remaining_tokens = len(tokens) - start

                if remaining_tokens < max_length and start != 0:
                    # Adjust start for the last window to ensure it has max_length tokens
                    start = max(0, len(tokens) - max_length)

                end = min(start + max_length, len(tokens))

                new_row = {}
                new_row["document"] = row["document"]
                new_row["valid"] = row["valid"]
                new_row["tokens"] = tokens[start:end]
                new_row["trailing_whitespace"] = row["trailing_whitespace"][start:end]
                new_row["labels"] = row["labels"][start:end]
                new_row["token_indices"] = list(range(start, end))
                new_row["full_text"] = rebuild_text(
                    new_row["tokens"], new_row["trailing_whitespace"]
                )
                new_row["sourse"] = row["source"]
                new_df.append(new_row)

                if remaining_tokens >= max_length:
                    start += doc_stride
                else:
                    # Break the loop if we've adjusted for the last window
                    break
        else:
            new_row = {
                "document": row["document"],
                "valid": row["valid"],
                "tokens": row["tokens"],
                "trailing_whitespace": row["trailing_whitespace"],
                "labels": row["labels"],
                "token_indices": row["token_indices"],
                "full_text": row["full_text"],
                "source": row["source"],
            }
            new_df.append(new_row)

    return pd.DataFrame(new_df)


def tokenize_for_inference(example, tokenizer, max_length, stride):
    text = []
    token_map = []

    for idx, (t, ws) in enumerate(
        zip(example["tokens"], example["trailing_whitespace"])
    ):
        text.append(t)
        token_map.extend([idx] * len(t))

        if ws:
            text.append(" ")
            token_map.append(-1)

    tokenized = tokenizer(
        "".join(text),
        return_offsets_mapping=True,
        truncation=True,
        max_length=max_length,
        stride=stride,
        return_overflowing_tokens=True,
    )

    return {**tokenized, "token_map": token_map}