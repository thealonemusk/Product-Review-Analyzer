import io

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import pickle
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from main import labels

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'product-review-analysis-6ac4f.appspot.com'
})



app = FastAPI()


class ReviewItem(BaseModel):
    Review: str


class SentimentPrediction(BaseModel):
    sentiment: str


# Load the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Load the trained model
# model_path = "./sentiment_model.pkl"
# with open(model_path, 'rb') as f:
#     model = pickle.load(f)
file_name = 'sentiment_model.pkl'
bucket = storage.bucket()
blob = bucket.blob(file_name)

print('downloading model')
content = blob.download_as_bytes()
print(content)
content_file = io.BytesIO(content)
model = pickle.load(content_file)


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
