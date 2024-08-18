from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model,TaskType
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "wnli"]
# task = "stsb"

model_checkpoint = "bert-base-uncased"
batch_size = 16
rank=4
for task in GLUE_TASKS:
    print("---------------------")
    print("processing -----",task)
    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    metric = load_metric('glue', actual_task)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)


    #--------------- PREPROCESS
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
            return tokenizer(examples[sentence1_key], truncation=True)
        return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    #----------------- MODEL and LoRA
    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2

    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels).to(device)

    for param in model.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS,r=rank, lora_alpha=32, lora_dropout=0.1, target_modules=['query', 'key', 'value'])
    model = get_peft_model(model, lora_config).to(device)

    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False

    print(model.print_trainable_parameters())

    #--------------------TRAINER
    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"

    model_name = model_checkpoint.split("/")[-1]

    args = TrainingArguments(
        f"{model_name}-finetuned-{task}",
        eval_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        # push_to_hub=True,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        return metric.compute(predictions=predictions, references=labels)

    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"
    trainer = Trainer(
        model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    #--------------------------SAVE MODEL
    model.save_pretrained(f'Bert_lora_{task}')