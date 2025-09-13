import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Load your dataset (career_data.csv) with columns: description, career
df = pd.read_csv("career_data.csv")
X = df['description'].tolist()
y = df['career'].tolist()

# Encode career labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Save label encoder for later use
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
encodings = tokenizer(X, truncation=True, padding=True, return_tensors='pt')
labels = torch.tensor(y_enc)

# Dataset class
class CareerDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

dataset = CareerDataset(encodings, labels)

# Initialize BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(le.classes_))

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_dir='./logs',
    logging_steps=10
)

# Trainer
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

# Save fine-tuned model
model.save_pretrained("career_bert_model")
tokenizer.save_pretrained("career_bert_model")

print("BERT fine-tuning completed. Model saved in 'career_bert_model'")
