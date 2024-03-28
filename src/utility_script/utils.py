# https://www.kaggle.com/code/valentinwerner/915-deberta3base-inference?scriptVersionId=161126788
# https://www.kaggle.com/code/sinchir0/visualization-code-using-displacy

import os
import json
import numpy as np
import pandas as pd
import wandb
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm.auto import tqdm
from spacy.tokens import Span
from spacy import displacy
import argparse
from ast import literal_eval
from transformers import Trainer
from torch.nn import CrossEntropyLoss


def read_json(file_path: str):
    fp = open(file_path)
    return json.load(fp)


def write_json_file(data, file_name):
    with open(file_name, "w") as fp:
        json.dump(data, fp)


def convert_to_json(data):
    return [
        {
            "document": row["document"],
            "full_text": row["full_text"],
            "tokens": row["tokens"],
            "trailing_whitespace": row["trailing_whitespace"],
            "labels": row["labels"],
            "valid": row["valid"],
            "source": row["source"],
            "token_indices": row["token_indices"],
        }
        for _, row in data.iterrows()
    ]


def str2bool(v):
    "Fix Argparse to process bools"
    if isinstance(v, bool):
        return v
    if v.lower() == "true":
        return True
    elif v.lower() == "false":
        return False

    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args(config):
    print("Running with the following config")
    parser = argparse.ArgumentParser(description="Run training baseline")
    for k, v in config.__dict__.items():
        parser.add_argument(
            "--" + k,
            type=type(v) if type(v) is not bool else str2bool,
            default=v,
            help=f"Default: {v}",
        )
    args = vars(parser.parse_args())

    # update config with parsed args
    for k, v in args.items():
        try:
            # attempt to eval it it (e.g. if bool, number, or etc)
            attempt = literal_eval(v)
        except (SyntaxError, ValueError):
            # if that goes wrong, just use the string
            attempt = v
        setattr(config, k, attempt)
        print(f"--{k}:{v}")


def parse_predictions(predictions, id2label, ds, threshold=0.9, softmax=True):
    if softmax:
        pred_softmax = np.exp(predictions) / np.sum(
            np.exp(predictions), axis=2
        ).reshape(predictions.shape[0], predictions.shape[1], 1)
    else:
        pred_softmax = preds.copy()

    preds = predictions.argmax(-1)
    preds_without_O = pred_softmax[:, :, :12].argmax(-1)
    O_preds = pred_softmax[:, :, 12]
    preds_final = np.where(O_preds < threshold, preds_without_O, preds)

    pairs = set()
    row, document, token, label, token_str = [], [], [], [], []
    for i, (p, token_map, offsets, tokens, doc, indices) in enumerate(
        zip(
            preds_final,
            ds["token_map"],
            ds["offset_mapping"],
            ds["tokens"],
            ds["document"],
            ds["token_indices"],
        )
    ):

        for token_pred, (start_idx, end_idx) in zip(p, offsets):
            label_pred = id2label[token_pred]

            if start_idx + end_idx == 0:
                continue

            if token_map[start_idx] == -1:
                start_idx += 1

            # ignore "\n\n"
            while start_idx < len(token_map) and tokens[token_map[start_idx]].isspace():
                start_idx += 1

            if start_idx >= len(token_map):
                break

            original_token_id = token_map[start_idx]
            token_id = indices[original_token_id]

            # ignore "O" predictions and whitespace preds
            if label_pred != "O" and token_id != -1:
                pair = (doc, token_id)

                if pair not in pairs:
                    row.append(i)
                    document.append(doc)
                    token.append(token_id)
                    label.append(label_pred)
                    token_str.append(tokens[original_token_id])
                    pairs.add(pair)

    df = pd.DataFrame(
        {
            "eval_row": row,
            "document": document,
            "token": token,
            "label": label,
            "token_str": token_str,
        }
    )

    df = df.drop_duplicates().reset_index(drop=True)

    df["row_id"] = list(range(len(df)))
    return df


# Mapping of named colors to their RGB equivalents
named_color_to_rgb = {
    "aqua": (0, 255, 255),
    "skyblue": (135, 206, 235),
    "limegreen": (50, 205, 50),
    "lime": (0, 255, 0),
    "hotpink": (255, 105, 180),
    "lightpink": (255, 182, 193),
    "purple": (128, 0, 128),
    "rebeccapurple": (102, 51, 153),
    "red": (255, 0, 0),
    "salmon": (250, 128, 114),
    "silver": (192, 192, 192),
    "lightgray": (211, 211, 211),
    "brown": (165, 42, 42),
    "chocolate": (210, 105, 30),
    # Add the rest of your named colors and their RGB values here
}


def get_rgba(color_name, opacity):
    """Convert a named color and opacity to an rgba string."""
    rgb = named_color_to_rgb[color_name]  # Get the RGB values for the named color
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {opacity})"


def visualize_prediction_html(
    tokenizer, row_id, predictions, id2label, ds, threshold=0.7
):
    # Define colors for each label using the provided color map
    label_colors = {
        "B-NAME_STUDENT": "aqua",
        "I-NAME_STUDENT": "skyblue",
        "B-EMAIL": "limegreen",
        "I-EMAIL": "lime",
        "B-USERNAME": "hotpink",
        "I-USERNAME": "lightpink",
        "B-ID_NUM": "purple",
        "I-ID_NUM": "rebeccapurple",
        "B-PHONE_NUM": "red",
        "I-PHONE_NUM": "salmon",
        "B-URL_PERSONAL": "silver",
        "I-URL_PERSONAL": "lightgray",
        "B-STREET_ADDRESS": "brown",
        "I-STREET_ADDRESS": "chocolate",
    }

    # Process predictions to get softmax probabilities, excluding the last column ("O" class)
    pred_softmax = np.exp(predictions[:, :, :-1]) / np.sum(
        np.exp(predictions[:, :, :-1]), axis=2, keepdims=True
    )

    # Find the document in the dataset using row_id
    doc_tokens = [tokenizer.decode(x) for x in ds["input_ids"][row_id]]
    doc_predictions = pred_softmax[row_id]
    doc_labels = doc_predictions.argmax(-1)

    # Start building HTML content
    html_content = '<div style="font-family: Arial; color: black; font-size: larger;">'
    for token, token_pred, token_softmax in zip(
        doc_tokens, doc_labels, doc_predictions
    ):
        label_pred = id2label[token_pred]
        if label_pred != "O":
            color = label_colors.get(
                label_pred, "#FFFFFF"
            )  # Default color is white if label not found
            opacity = np.max(token_softmax)  # Use the max softmax score as opacity
            rgba_color = get_rgba(color, opacity)
            html_content += (
                f'<span style="background-color: {rgba_color};">{token}</span> '
            )
        else:
            html_content += f"{token} "

    # Add all tokens with probability above threshold
    high_prob_indices = np.where(doc_predictions.max(-1) > threshold)
    for index in high_prob_indices[0]:
        high_prob_label = id2label[doc_labels[index]]
        high_prob_value = doc_predictions.max(-1)[index]
        high_prob_color = label_colors.get(high_prob_label, "#FFFFFF")
        rgba_color = get_rgba(high_prob_color, high_prob_value)
        html_content += f'<p>High probability token: <span style="background-color: {rgba_color};">{doc_tokens[index]}</span> Label: {high_prob_label} Probability: {high_prob_value}</p>'

    # Add legend for class-color mapping
    html_content += "<div><p>Legend:</p><ul>"
    for label, color in label_colors.items():
        html_content += f'<li><span style="background-color: {get_rgba(color, 1)};">{label}</span></li>'
    html_content += "</ul></div></div>"
    return html_content


def get_reference_df_parquet(artifact, filename="raw_data.parquet", valid=True):
    raw_artifact = wandb.use_artifact(artifact)
    raw_artifact_dir = raw_artifact.download()
    raw_df = pd.read_parquet(raw_artifact_dir + f"/{filename}")
    if valid:
        raw_df = raw_df[raw_df["valid"] == True].copy()

    ref_df = raw_df[["document", "tokens", "labels"]].copy()
    ref_df = (
        ref_df.explode(["tokens", "labels"])
        .reset_index(drop=True)
        .rename(columns={"tokens": "token", "labels": "label"})
    )
    ref_df["token"] = ref_df.groupby("document").cumcount()

    reference_df = ref_df[ref_df["label"] != "O"].copy()
    reference_df = reference_df.reset_index().rename(columns={"index": "row_id"})
    reference_df = reference_df[["row_id", "document", "token", "label"]].copy()

    return reference_df


def get_reference_df(valid):
    ref_df = valid[["document", "tokens", "labels"]].copy()
    ref_df = (
        ref_df.explode(["tokens", "labels"])
        .reset_index(drop=True)
        .rename(columns={"tokens": "token", "labels": "label"})
    )
    ref_df["token"] = ref_df.groupby("document").cumcount()

    reference_df = ref_df[ref_df["label"] != "O"].copy()
    reference_df = reference_df.reset_index().rename(columns={"index": "row_id"})
    reference_df = reference_df[["row_id", "document", "token", "label"]].copy()

    return reference_df


def process_html(tokenizer, index, predictions, id2label, valid_ds, threshold=0.9):
    return index, wandb.Html(
        visualize_prediction_html(
            tokenizer, index, predictions, id2label, valid_ds, threshold=0.9
        )
    )


def generate_htmls_concurrently(
    viz_df, tokenizer, predictions, id2label, valid_ds, threshold=0.9
):
    results_with_index = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_html,
                tokenizer,
                i,
                predictions,
                id2label,
                valid_ds,
                threshold=0.9,
            )
            for i in viz_df.index.values.tolist()
        ]
        for future in tqdm(as_completed(futures), total=len(viz_df)):
            results_with_index.append(future.result())

    # Sort the results by index to maintain the original order
    results_with_index.sort(key=lambda x: x[0])
    htmls = [result[1] for result in results_with_index]
    return htmls


def visualize(row, nlp):
    options = {
        "colors": {
            "B-NAME_STUDENT": "aqua",
            "I-NAME_STUDENT": "skyblue",
            "B-EMAIL": "limegreen",
            "I-EMAIL": "lime",
            "B-USERNAME": "hotpink",
            "I-USERNAME": "lightpink",
            "B-ID_NUM": "purple",
            "I-ID_NUM": "rebeccapurple",
            "B-PHONE_NUM": "red",
            "I-PHONE_NUM": "salmon",
            "B-URL_PERSONAL": "silver",
            "I-URL_PERSONAL": "lightgray",
            "B-STREET_ADDRESS": "brown",
            "I-STREET_ADDRESS": "chocolate",
        }
    }
    doc = nlp(row.full_text)
    doc.ents = [
        Span(doc, idx, idx + 1, label=label)
        for idx, label in enumerate(row.labels)
        if label != "O"
    ]
    html = displacy.render(doc, style="ent", jupyter=False, options=options)
    return html


def convert_for_upload(viz_df):
    mapping = {
        "index": "str",
        "document_x": "str",
        "valid": "str",
        "tokens": "str",
        "trailing_whitespace": "str",
        "labels": "str",
        "token_indices": "str",
        "full_text": "str",
        "unique_labels": "str",
        "EMAIL": "str",
        "ID_NUM": "str",
        "NAME_STUDENT": "str",
        "PHONE_NUM": "str",
        "STREET_ADDRESS": "str",
        "URL_PERSONAL": "str",
        "USERNAME": "str",
        "NO_PII": "str",
        "document_y": "str",
        "token": "str",
        "label": "str",
        "token_str": "str",
    }
    for key, type in mapping.items():
        viz_df[key] = viz_df[key].astype(type)
    return viz_df


def filter_errors(eval_df, preds_df):
    target_strings = []
    for i, row in eval_df.iterrows():
        target_string = [
            f"{t}: {l}" for t, l in zip(row.tokens, row.labels) if l != "O"
        ]
        target_strings.append(" ".join(target_string))

    pred_strings = []
    for i in range(len(eval_df)):
        i_preds = preds_df[preds_df.eval_row == i]
        if len(i_preds) > 0:
            pred_string = [
                f"{t}: {l}" for t, l in zip(i_preds.token_str, i_preds.label)
            ]
        else:
            pred_string = []
        pred_strings.append(" ".join(pred_string))

    eval_df["target_string"] = target_strings
    eval_df["pred_string"] = pred_strings
    eval_df["error"] = eval_df["target_string"] != eval_df["pred_string"]
    return eval_df[eval_df.error == True]


class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Assuming class_weights is a Tensor of weights for each class
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        # Extract labels
        labels = inputs.pop("labels")

        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Reshape for loss calculation
        loss_fct = CrossEntropyLoss(weight=self.class_weights)

        if self.label_smoother is not None and "labels" in inputs:
            loss = self.label_smoother(outputs, inputs)
        else:
            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels), labels.view(-1)
            )

        return (loss, outputs) if return_outputs else loss


# https://www.kaggle.com/competitions/llm-detect-ai-generated-text/discussion/465970#2589607
def upload_kaggle_dataset(storage_dir, dataset_name, owner="thedrcat"):
    """
    :param storage_dir: upload storage dir to kaggle as dataset
    :param dataset_name: name of the dataset
    :param owner: name of the dataset owner
    """
    print(f"creating metadata...")
    os.system(f"kaggle datasets init -p {storage_dir}")

    print(f"updating metadata...")
    with open(os.path.join(storage_dir, "dataset-metadata.json"), "r") as f:
        metadata = json.load(f)

    metadata["title"] = dataset_name
    metadata["id"] = f"{owner}/{dataset_name}".replace("_", "-")

    print(f"saving updated metadata...")
    with open(os.path.join(storage_dir, "dataset-metadata.json"), "w") as f:
        json.dump(metadata, f)

    print("uploading the dataset ...")
    os.system(f"kaggle datasets create -p {storage_dir}")
    print("done!")