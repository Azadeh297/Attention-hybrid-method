from datasets import load_dataset
from evaluate import load as load_evaluate
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

task = "ag_news"
model_checkpoint = "bert-base-uncased"
batch_size = 16
d_model = 768

dataset = load_dataset("ag_news")
metric = load_evaluate('accuracy')

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)

task_to_keys = {
    "ag_news": ("text", None),
    "cola": ("sentence", None),
    "imdb": ("sentence", None),
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
    return tokenizer(examples[sentence1_key], truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)


num_labels = 4  

class CombinedModel(nn.Module):
    def __init__(self, base_model, num_labels):
        super(CombinedModel, self).__init__()
        self.base_model = base_model
        self.feed_forward = nn.Linear(d_model, d_model)
        self.classifier = nn.Linear(768, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn_r = nn.MSELoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs4 = self.base_model(input_ids=input_ids, attention_mask=attention_mask).hidden_states[-1][:, 0, :]
        outputs_task4 = self.feed_forward(outputs4)
        final_output = outputs_task4
        logits = self.classifier(final_output)

        if labels is not None:
            if task == "stsb":
                loss = self.loss_fn_r(logits, labels)
            else:
                loss = self.loss_fn(logits, labels)
            return loss, logits
        
        return logits

base_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels, output_hidden_states=True).to(device)
for param in base_model.parameters():
    param.requires_grad = False

combined_model = CombinedModel(base_model, num_labels).to(device)

metric_name = "accuracy"
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-FFN_finetuned-{task}",
    eval_strategy="epoch",
    save_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=False,
    metric_for_best_model=metric_name,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    combined_model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],  
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
