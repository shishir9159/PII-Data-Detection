import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
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

    return df
