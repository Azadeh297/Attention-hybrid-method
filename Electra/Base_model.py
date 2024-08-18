from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2","stsb", "wnli"]
# GLUE_TASKS = ["wnli"]
task = "stsb"


for task in GLUE_TASKS:
    
    # model_checkpoint = "google/electra-base-discriminator"
    model_checkpoint = "google/electra-small-discriminator"
    
    batch_size = 16

    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    metric = load_metric('glue', actual_task,  trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

    # Preprocess the dataset
    task_to_keys = {
        "cola": ("sentence", None),
        "mnli": ("premise", "hypothesis"),
        "mnli-mm": ("premise", "hypothesis"),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
        "qqp": ("question1", "question2"),
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "stsb": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2"),
    }

    sentence1_key, sentence2_key = task_to_keys[task]

    def preprocess_function(examples):
        if sentence2_key is None:
            return tokenizer(examples[sentence1_key], truncation=True, padding="max_length", max_length=512)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True, padding="max_length", max_length=512)

    print(f"Preprocessing {task} dataset...")
    encoded_dataset = dataset.map(preprocess_function, batched=True)

    # Remove unnecessary columns but keep the labels
    valid = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"

    columns_to_remove = list(set(encoded_dataset[valid].column_names) - {"input_ids", "attention_mask", "label"})
    encoded_dataset = encoded_dataset.remove_columns(columns_to_remove)



    # Load pre-trained model
    num_labels = 3 if task.startswith("mnli") else 1 if task == "stsb" else 2
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels).to(device)
    model.eval()

    # Data Collator for padding
    data_collator = DataCollatorWithPadding(tokenizer)

    # DataLoader for batching
    eval_dataset = encoded_dataset["validation_mismatched"] if task == "mnli-mm" else encoded_dataset["validation_matched"] if task == "mnli" else encoded_dataset["validation"]
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False)

    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
    # Inference
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    predictions, labels = [], []
    for batch in tqdm(eval_loader, desc="Evaluating "+task):
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
            outputs = model(**inputs)
            logits = outputs.logits
            predictions.append(logits.cpu().numpy())
            # print("batch= ",batch)
            labels.append(batch["labels"].cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)
    labels = np.concatenate(labels, axis=0)

    metrics = compute_metrics((predictions, labels))
    print(metrics)
