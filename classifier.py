import pandas as pd
import torch
import gc
from dataset import Dataset
from sklearn.preprocessing import LabelEncoder
import huggingface_hub
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification,
                          DataCollatorWithPadding,
                          TrainingArguments,
                          Trainer,
                          pipeline,
                        )

class IntentClassifier:

   def __init__(self):
      self.model_name = "distilbert/distilbert-base-multilingual-cased" 
   
   def train_model(self, train_data, test_data):
       
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name,
                                                                   num_labels=self.num_labels,
                                                                   id2label=self.label_dict)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        training_args = TrainingArguments(
                    output_dir = self.model_path,
                    learning_rate=2e-4,
                    per_device_train_batch_size=32,
                    per_device_eval_batch_size=32,
                    num_train_epochs=30,
                    weight_decay=0.01,
                    evaluation_strategy="epoch",
                    logging_strategy="epoch",
                    report_to="None",
                    push_to_hub=True,
                )
        
        trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset = train_data,
                    eval_dataset = test_data,
                    tokenizer = self.tokenizer,
                    data_collator=data_collator,
                    compute_metrics= compute_metrics
                )

        trainer.set_device(self.device)

        trainer.train()

        # Flush Memory
        del trainer,model
        gc.collect()

        if self.device == 'cuda':
            torch.cuda.empty_cache()
