import pandas as pd
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
import pickle

print('Training')

# Load and preprocess the dataset
dataset = pd.read_csv("testing_dataset.csv")

# Split the dataset into training and validation sets
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Define the labels (sentiments)
labels = list(dataset["Rate"].unique())

# Load the pre-trained tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenize the text and encode labels
train_encodings = tokenizer(list(train_data["Review"]), truncation=True, padding=True)
val_encodings = tokenizer(list(val_data["Review"]), truncation=True, padding=True)

train_input_ids = torch.tensor(train_encodings["input_ids"])
train_attention_mask = torch.tensor(train_encodings["attention_mask"])
train_labels = torch.tensor([labels.index(label) for label in train_data["Rate"]])

val_input_ids = torch.tensor(val_encodings["input_ids"])
val_attention_mask = torch.tensor(val_encodings["attention_mask"])
val_labels = torch.tensor([labels.index(label) for label in val_data["Rate"]])

# Create PyTorch DataLoader
train_dataset = torch.utils.data.TensorDataset(train_input_ids, train_attention_mask, train_labels)
val_dataset = torch.utils.data.TensorDataset(val_input_ids, val_attention_mask, val_labels)

# Load the pre-trained model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(labels))

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.train()

num_epochs = 3
batch_size = 16

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss}")

# Save the trained model as a .pkl file
model_path = "./sentiment_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

# Save the tokenizer
tokenizer.save_pretrained("./sentiment_model")
