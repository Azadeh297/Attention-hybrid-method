from datasets import load_dataset, load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model,TaskType
import numpy as np
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2",  "wnli"]


#------------------------------------------------------------------------------------------
#------------------TASK1
model_dir = 'El_base_lora_cola'
model_task1 = AutoModelForSequenceClassification.from_pretrained(model_dir, output_hidden_states=True).to(device)
for param in model_task1.parameters():
    param.requires_grad = False
#------------------TASK2
model_dir = 'El_base_lora_mnli-mm'
model_task2 = AutoModelForSequenceClassification.from_pretrained(model_dir,num_labels=3, output_hidden_states=True).to(device)
for param in model_task2.parameters():
    param.requires_grad = False
#------------------TASK3
model_dir = 'El_base_lora_mnli'
model_task3 = AutoModelForSequenceClassification.from_pretrained(model_dir,num_labels=3, output_hidden_states=True).to(device)
for param in model_task3.parameters():
    param.requires_grad = False
#------------------TASK4
model_dir = 'El_base_lora_mrpc'
model_task4 = AutoModelForSequenceClassification.from_pretrained(model_dir, output_hidden_states=True).to(device)
for param in model_task4.parameters():
    param.requires_grad = False
#------------------TASK5
model_dir = 'El_base_lora_qnli'
model_task5 = AutoModelForSequenceClassification.from_pretrained(model_dir, output_hidden_states=True).to(device)
for param in model_task5.parameters():
    param.requires_grad = False
#------------------TASK6
model_dir = 'El_base_lora_wnli'
model_task6 = AutoModelForSequenceClassification.from_pretrained(model_dir, output_hidden_states=True).to(device)
for param in model_task6.parameters():
    param.requires_grad = False
#------------------TASK7
model_dir = 'El_base_lora_rte'
model_task7 = AutoModelForSequenceClassification.from_pretrained(model_dir, output_hidden_states=True).to(device)
for param in model_task7.parameters():
    param.requires_grad = False
#------------------TASK8
model_dir = 'El_base_lora_sst2'
model_task8 = AutoModelForSequenceClassification.from_pretrained(model_dir, output_hidden_states=True).to(device)
for param in model_task8.parameters():
    param.requires_grad = False
#------------------TASK9
model_dir = 'El_base_lora_qqp'
model_task9 = AutoModelForSequenceClassification.from_pretrained(model_dir, output_hidden_states=True).to(device)
for param in model_task9.parameters():
    param.requires_grad = False

for task in GLUE_TASKS:
    print("--------------------------")
    print("proccessing ",task)
   
    model_checkpoint = "google/electra-base-discriminator"
    # model_checkpoint = "google/electra-small-discriminator"
    
    batch_size = 1
    rank=4
    d_model=768
    hidden_size=768

    actual_task = "mnli" if task == "mnli-mm" else task
    dataset = load_dataset("glue", actual_task)
    metric = load_metric('glue', actual_task,trust_remote_code=True)

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


    num_labels = 3 if task.startswith("mnli") else 1 if task=="stsb" else 2


    


    #-----------------------------------------------------------------------------------------------
    #----------------------------------------combine model -----------------------------------------
    class CombinedModel(nn.Module):
        def __init__(self, base_model, model1, model2, model3,model4, model5, model6,model7, model8, model9, num_labels,hidden_size):
            super(CombinedModel, self).__init__()
            self.base_model = base_model
            self.model1 = model1
            self.model2 = model2
            self.model3 = model3
            self.model4 = model4
            self.model5 = model5
            self.model6 = model6
            self.model7 = model7
            self.model8 = model8
            self.model9 = model9
            self.model=[self.model1,self.model2,self.model3,self.model4,self.model5,self.model6,self.model7,self.model8,self.model9]

            self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=1)
            self.layer_norm = nn.LayerNorm(hidden_size)
            self.feed_forward = nn.Linear(d_model, d_model)
            
            self.classifier = nn.Linear(d_model, num_labels)
            self.loss_fn = nn.CrossEntropyLoss()  
            self.loss_fn_r = nn.MSELoss()  # Mean Squared Error Loss

        def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
            
            #----------------------------------------------------------------- all token and agg them
            output=list()
            output_pool=list()
            for model in self.model:
                output.append(model(input_ids, attention_mask, token_type_ids).hidden_states[-1])
            for i in range(len(output)):
                output_pool.append(self.layer_norm(torch.mean(output[i], dim=1)))   # Agg all output of each model
            combined_outputs = torch.stack(output_pool, dim=1)
            print(combined_outputs.size())

            combined_outputs = combined_outputs.transpose(0, 1)
            attn_output,  attn_weights = self.attention(combined_outputs, combined_outputs, combined_outputs)
            attn_output = attn_output.transpose(0, 1)
            print(attn_output.size())
            print(attn_weights.shape)
            
            # Aggregate outputs
            agg_output = self.layer_norm(torch.mean(attn_output, dim=1))
            outputs4 = self.base_model(input_ids=input_ids, attention_mask=attention_mask).hidden_states[-1]
            outputs4_pooled = self.layer_norm(torch.mean(outputs4, dim=1))


            outputs_task4 = self.feed_forward(outputs4_pooled)
            # print("outputs_task4= ",outputs_task4.size())


            final_output=self.layer_norm(outputs_task4 + agg_output)

            
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

    combined_model = CombinedModel(base_model, model_task1,model_task2, model_task3,
                                model_task4,model_task5, model_task6,
                                model_task7,model_task8, model_task9,
                                    num_labels,hidden_size).to(device)



    #--------------------TRAINER
    metric_name = "pearson" if task == "stsb" else "matthews_correlation" if task == "cola" else "accuracy"
    model_name = model_checkpoint.split("/")[-1]

    args = TrainingArguments(
        f"{model_name}-combine_finetuned-{task}",
        eval_strategy = "epoch",
        # save_strategy = "epoch",
        save_strategy="no", 
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=20,
        weight_decay=0.01,
        load_best_model_at_end=False,
        metric_for_best_model=metric_name,
        # push_to_hub=True,
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(predictions, axis=1)
        else:
            predictions = predictions[:, 0]
        # return metric.compute(predictions=predictions, references=labels)

        metrics = metric.compute(predictions=predictions, references=labels)

        # Save current metrics to a file
        with open("current_metrics.txt", "w") as f:
            f.write(str(metrics))

        return metrics

    validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"



    from transformers import TrainerCallback, TrainerState, TrainerControl

    class SaveBestMetricCallback(TrainerCallback):
        def __init__(self, metric_name, output_file):
            self.metric_name = metric_name
            self.output_file = output_file
            self.best_metric_value = None

        def on_evaluate(self, args, state, control, **kwargs):
            # Iterate through log history to find the most recent evaluation metric
            current_metric_value = None
            for log in reversed(state.log_history):
                if self.metric_name in log:
                    current_metric_value = log[self.metric_name]
                    break

            # Update and save the best metric value
            if current_metric_value is not None:
                if self.best_metric_value is None or current_metric_value > self.best_metric_value:
                    self.best_metric_value = current_metric_value
                    with open(self.output_file, "w") as f:
                        f.write(f"Best {self.metric_name}: {self.best_metric_value}\n")

    # Example usage
    save_best_metric_callback = SaveBestMetricCallback(metric_name, "best_metric.txt")

    trainer = Trainer(
        combined_model,
        args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset[validation_key],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[save_best_metric_callback]
    )

    trainer.train()
  