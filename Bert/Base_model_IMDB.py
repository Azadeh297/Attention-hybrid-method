from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_checkpoint = "bert-base-uncased"
batch_size = 16

# Load the IMDB dataset
dataset = load_dataset("imdb")
metric = load_metric('accuracy')

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)

print("Preprocessing IMDB dataset...")
encoded_dataset = dataset.map(preprocess_function, batched=True)
# print("encoded_dataset",encoded_dataset)

# Remove unnecessary columns but keep the labels
columns_to_remove = list(set(encoded_dataset["train"].column_names) - set(['input_ids', 'attention_mask', 'label']))
encoded_dataset = encoded_dataset.remove_columns(columns_to_remove)
# print("encoded_dataset",encoded_dataset)

data_collator = DataCollatorWithPadding(tokenizer)

eval_dataloader = DataLoader(encoded_dataset["test"], batch_size=batch_size, collate_fn=data_collator)

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
model.to(device)

# Evaluation
model.eval()
all_predictions = []
all_labels = []

for batch in tqdm(eval_dataloader):
    inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
    labels = batch['labels'].to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    all_predictions.extend(predictions.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

eval_metric = metric.compute(predictions=all_predictions, references=all_labels)
print(f"Accuracy: {eval_metric['accuracy']}")
