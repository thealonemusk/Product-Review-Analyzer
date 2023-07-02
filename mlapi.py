from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import pickle

from main import labels

app = FastAPI()


class ReviewItem(BaseModel):
    Review: str


class SentimentPrediction(BaseModel):
    sentiment: str


# Load the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Load the trained model
model_path = "./sentiment_model.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)


@app.post('/predict', response_model=SentimentPrediction)
async def predict_sentiment(item: ReviewItem):
    # Tokenize the input text
    inputs = tokenizer(item.Review, truncation=True, padding=True, return_tensors='pt')
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    # Make predictions
    outputs = model(input_ids, attention_mask)
    predicted_labels = torch.argmax(outputs.logits, dim=1)

    # Map the predicted label to sentiment
    sentiment = labels[predicted_labels.item()]

    # Create the response
    prediction = SentimentPrediction(sentiment=sentiment)
    return prediction
