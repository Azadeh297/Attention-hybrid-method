from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model,TaskType
import numpy as np
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2","stsb",  "wnli"]
task = "stsb"
model_checkpoint = "bert-base-uncased"
batch_size = 16
rank=4
d_model=768

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
# if sentence2_key is None:
#     print(f"Sentence: {dataset['train'][0][sentence1_key]}")
# else:
#     print(f"Sentence 1: {dataset['train'][0][sentence1_key]}")
#     print(f"Sentence 2: {dataset['train'][0][sentence2_key]}")

def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)

#----------------- MODEL and LoRA
num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2


#-----------------------------------------------------------------------------------------------

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


        final_output=outputs_task4 

        
        logits = self.classifier(final_output)

        if labels is not None:
            if task=="stsb":
                loss=self.loss_fn_r(logits,labels)
            else:
                loss = self.loss_fn(logits, labels)
            return loss, logits
        
        return logits


base_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels, output_hidden_states=True).to(device)
for param in base_model.parameters():
    param.requires_grad = False

combined_model = CombinedModel(base_model, num_labels).to(device)



#--------------------TRAINER
metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
model_name = model_checkpoint.split("/")[-1]

args = TrainingArguments(
    f"{model_name}-FFN_finetuned-{task}",
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
    combined_model,
    args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset[validation_key],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
#--------------------------SAVE MODEL
# combined_model.save_pretrained(f'combine_{task}')