from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_checkpoint = "bert-base-uncased"
batch_size = 16
rank = 4

# Load the IMDB dataset
dataset = load_dataset("imdb")
metric = load_metric('accuracy')

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)

print("Preprocessing IMDB dataset...")
encoded_dataset = dataset.map(preprocess_function, batched=True)

# Remove unnecessary columns but keep the labels
columns_to_remove = list(set(encoded_dataset["train"].column_names) - set(['input_ids', 'attention_mask', 'label']))
encoded_dataset = encoded_dataset.remove_columns(columns_to_remove)

# Model and LoRA configuration
num_labels = 2

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels).to(device)

for param in model.parameters():
    param.requires_grad = False

lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=32, lora_dropout=0.1, target_modules=['query', 'key', 'value'])
model = get_peft_model(model, lora_config).to(device)

for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=lambda p: metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
)

# Fine-tune the model
trainer.train()


