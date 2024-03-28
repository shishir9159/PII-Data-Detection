import os
import torch
import wandb
import spacy
from itertools import chain
from functools import partial
from transformers import AutoTokenizer, TrainingArguments
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)
import pandas as pd
from types import SimpleNamespace

# Import necessary functions and classes from other files
from metric import compute_metrics
from data import create_dataset
from utils import get_reference_df, parse_predictions
from utils import (
    filter_errors,
    generate_htmls_concurrently,
    visualize,
    convert_for_upload,
)

from utils import parse_args
from utils import CustomTrainer

# Define the project name for Weights & Biases
WANDB_PROJECT = "Kaggle-PII"
USER_NAME = "nahidhasan43"
PROJECT_PATH = f"{USER_NAME}/{WANDB_PROJECT}"
EXPERIMENT = "longformerForPiiDetection"

# Define the configuration for the experiment
config = SimpleNamespace(
    experiment=EXPERIMENT,
    threshold=0.6,
    o_weight=0.05,
    stride_artifact=f"{PROJECT_PATH}/processed_data:latest",
    raw_artifact=f"{PROJECT_PATH}/raw_data:latest",
    external_data_1="none",
    external_data_2="none",
    external_data_3="none",
    external_data_4="none",
    external_data_5="none",
    output_dir="output",
    inference_max_length=768,
    training_max_length=512,
    training_model_path="allenai/longformer-base-4096",
    fp16=True,
    learning_rate=1e-5,
    num_train_epochs=0.1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    report_to="wandb",
    evaluation_strategy="epoch",
    do_eval=True,
    save_total_limit=1,
    logging_steps=10,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
)


def main(config):
    # Initialize Weights & Biases
    wandb.init(project=WANDB_PROJECT, job_type="train", config=config)
    config = wandb.config

    # Load the data
    stride_artifact = wandb.use_artifact(config.stride_artifact)
    stride_artifact_dir = stride_artifact.download()
    df = pd.read_parquet(stride_artifact_dir + "/stride_data.parquet")
    train_df = df[df.valid == False].reset_index(drop=True)
    eval_df = df[df.valid == True].reset_index(drop=True)

    # Load external data
    for art in [
        config.external_data_1,
        config.external_data_2,
        config.external_data_3,
        config.external_data_4,
        config.external_data_5,
    ]:
        if art != "none":
            print(f"Loading external data {art}...")
            artifact = wandb.use_artifact(art)
            artifact_dir = artifact.download()
            ext_df = pd.read_parquet(artifact_dir + "/ext_data.parquet")
            train_df = pd.concat([train_df, ext_df], ignore_index=True)

    # Prepare references and labels
    reference_df = get_reference_df(config.raw_artifact)
    all_labels = sorted(list(set(chain(*[x.tolist() for x in df.labels.values]))))
    label2id = {l: i for i, l in enumerate(all_labels)}
    id2label = {v: k for k, v in label2id.items()}

    # Create the training and validation datasets
    tokenizer = AutoTokenizer.from_pretrained(config.training_model_path)
    train_ds = create_dataset(train_df, tokenizer, config.training_max_length, label2id)
    valid_ds = create_dataset(eval_df, tokenizer, config.inference_max_length, label2id)

    # Initialize the model and data collator
    model = AutoModelForTokenClassification.from_pretrained(
        config.training_model_path,
        num_labels=len(all_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

    # Define the training arguments
    args = TrainingArguments(
        output_dir=config.output_dir,
        fp16=config.fp16,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        report_to=config.report_to,
        evaluation_strategy=config.evaluation_strategy,
        do_eval=config.do_eval,
        save_total_limit=config.save_total_limit,
        logging_steps=config.logging_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
    )

    # Calculate class weights based on your dataset (TODO: move to config)
    class_weights = torch.tensor([1.0] * 12 + [config.o_weight]).to("cuda")

    # Initialize Trainer with custom class weights
    trainer = CustomTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=partial(
            compute_metrics,
            id2label=id2label,
            valid_ds=valid_ds,
            valid_df=reference_df,
            threshold=config.threshold,
        ),
        class_weights=class_weights,
    )

    # Train the model
    trainer.train()

    # Make predictions on the validation dataset
    preds = trainer.predict(valid_ds)

    # Compute the final metrics and log them to Weights & Biases
    print("Computing final metrics...")
    final_metrics = {
        f"final_f5_at_{threshold}": compute_metrics(
            (preds.predictions, None),
            id2label,
            valid_ds,
            reference_df,
            threshold=threshold,
        )["ents_f5"]
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97]
    }
    wandb.log(final_metrics)
    print(final_metrics)

    # pick the best threshold from the final metrics and use it to generate preds_df
    best_threshold = float(max(final_metrics, key=final_metrics.get).split("_")[-1])
    wandb.config.best_threshold = best_threshold
    preds_df = parse_predictions(
        preds.predictions, id2label, valid_ds, threshold=best_threshold
    )

    # Prepare data to visualize errors and log them as a Weights & Biases table
    print("Visualizing errors...")
    grouped_preds = preds_df.groupby("eval_row")[
        ["document", "token", "label", "token_str"]
    ].agg(list)
    viz_df = pd.merge(
        eval_df.reset_index(),
        grouped_preds,
        how="left",
        left_on="index",
        right_on="eval_row",
    )
    viz_df = filter_errors(viz_df, preds_df)
    viz_df["pred_viz"] = generate_htmls_concurrently(
        viz_df,
        tokenizer,
        preds.predictions,
        id2label,
        valid_ds,
        threshold=best_threshold,
    )
    nlp = spacy.blank("en")
    htmls = [visualize(row, nlp) for _, row in viz_df.iterrows()]
    wandb_htmls = [wandb.Html(html) for html in htmls]
    viz_df["gt_viz"] = wandb_htmls
    viz_df.fillna("", inplace=True)
    viz_df = convert_for_upload(viz_df)
    errors_table = wandb.Table(dataframe=viz_df)
    wandb.log({"errors_table": errors_table})

    # Save the model and upload it to Kaggle
    os.makedirs(config.experiment, exist_ok=True)
    trainer.save_model(config.experiment)
    # if training on a local machine, uncomment and fill in your username to upload the model to Kaggle
    # upload_kaggle_dataset(config.experiment, config.experiment, owner="thedrcat")
    print("Experiment finished, test it out on the inference notebook!")


if __name__ == "__main__":
    parse_args(config)
    main(config)