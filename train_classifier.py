import pandas as pd
import numpy as np
import torch
import gc
import os
from dotenv import load_dotenv
from dataset import Dataset
from utils import compute_metrics
from sklearn.preprocessing import LabelEncoder
import huggingface_hub
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, 
                          AutoModelForSequenceClassification,
                          DataCollatorWithPadding,
                          TrainingArguments,
                          Trainer,
                        )

load_dotenv()
                    

class IntentClassifier:

    def __init__(self):
       self.model_name = "distilbert/distilbert-base-multilingual-cased" 
       self.model_path = "tvcommand/intent-classifier-multilingual"
       self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
       self.huggingface_token = os.getenv("huggingface_token")
       self.num_labels = 5

       if self.huggingface_token is not None:
            huggingface_hub.login(self.huggingface_token)

       self.tokenizer = self.load_tokenizer()

       train_data, test_data = self.load_data(self.data_path)
       self.train_model(train_data, test_data)
   
    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return tokenizer
    

   
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

    def load_data(self):
        df = pd.read_csv("data\multilingual_intent_dataset.csv")

        # Encode Labels
        le = LabelEncoder()
        le.fit(df['label'].tolist())
       
        label_dict = {index:label_name for index, label_name in enumerate(le.__dict__['classes_'].tolist())}
        self.label_dict = label_dict
        df["label"] = le.fit_transform(df["label"].tolist())

        # Train / Test Split
        test_size = 0.1
        df_train, df_test = train_test_split(df, 
                                            test_size=test_size, 
                                            stratify=df['label'],)
        # Conver Pandas to a hugging face dataset
        train_dataset = Dataset.from_pandas(df_train)
        test_dataset = Dataset.from_pandas(df_test)

        train_dataset_tokenized = train_dataset.map(self.tokenize_func, batched=True)
        test_dataset_tokenized = test_dataset.map(self.tokenize_func, batched=True)

        return train_dataset_tokenized, test_dataset_tokenized

    def tokenize_func(self, examples):
        return self.tokenizer(examples['text'], truncation=True, padding='max_length')


if __name__ == "__main__":
    intent_classifier = IntentClassifier()    
    


    
