from datasets import DatasetDict, load_dataset
from evaluate import load as load_evaluate  
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_checkpoint = "bert-base-uncased"
batch_size = 16

# Load the TREC dataset 
dataset = DatasetDict({
    "train": load_dataset("trec", split="train"),
    "test": load_dataset("trec", split="test")
})

metric = load_evaluate('accuracy')  

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

# Preprocess the dataset
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=512)

print("Preprocessing TREC dataset...")
encoded_dataset = dataset.map(preprocess_function, batched=True)

# Remove unnecessary columns but keep the required ones
columns_to_keep = ['input_ids', 'attention_mask', 'coarse_label']
columns_to_remove = list(set(encoded_dataset["train"].column_names) - set(columns_to_keep))
encoded_dataset = encoded_dataset.remove_columns(columns_to_remove)

# Rename 'coarse_label' to 'labels' for compatibility with the model
encoded_dataset = encoded_dataset.rename_column("coarse_label", "labels")

data_collator = DataCollatorWithPadding(tokenizer)

eval_dataloader = DataLoader(encoded_dataset["test"], batch_size=batch_size, collate_fn=data_collator)

# The TREC dataset has 6 coarse labels
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=6)
model.to(device)

# Evaluation
model.eval()
all_predictions = []
all_labels = []

for batch in tqdm(eval_dataloader):
    inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
    labels = batch['labels'].to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    all_predictions.extend(predictions.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

eval_metric = metric.compute(predictions=all_predictions, references=all_labels)
print(f"Accuracy: {eval_metric['accuracy']}")
