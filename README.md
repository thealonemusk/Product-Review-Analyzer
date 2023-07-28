# Review Rocket

An LLM-powered API endpoint that analyzes product reviews and sentiment data to provide insights into product quality, customer satisfaction, and potential improvements. It can help businesses understand customer feedback and make data-driven decisions

## The Problem It Solves

Picture this: You're running a business, pouring your heart and soul into delivering exceptional products. But without a clear understanding of what your customers truly think, how can you make the right decisions to improve your offerings? Traditional methods of manual review analysis are time-consuming, inefficient, and often miss the mark. The problem is clear - businesses are drowning in a sea of reviews, and they need a lifeline to reach the shore of customer satisfaction.This API allows you to extract insights from product reviews and make judicious business decisions


## Designing and Training the model

The ML code consists of a sentiment classification model trained on customer reviews using the DistilBERT model. It performs the following steps:

-Loading and Preprocessing the Dataset: The code loads a dataset containing customer reviews from a CSV file. It splits the dataset into training and validation sets. The labels (sentiments) are extracted from the dataset.

-Tokenizing and Encoding: The DistilBERT tokenizer is loaded, and the text data is tokenized and encoded using the tokenizer. The labels are encoded as well.

-Creating PyTorch Datasets: PyTorch datasets are created using the encoded input and label data.

-Loading the Pre-trained Model: The pre-trained DistilBERT model for sequence classification is loaded with the appropriate number of labels.

-Defining the Optimizer: AdamW optimizer is defined to update the model's parameters during training.

-Training the Model: The model is trained for a specified number of epochs. The training loop iterates over batches of data, performs forward and backward passes, and updates the model's parameters using the optimizer.



## API documentation

The API code uses the FastAPI framework to deploy the sentiment classification model as a RESTful API. It exposes an endpoint for making predictions on new customer reviews. Here's how it works:

-FastAPI Setup: The FastAPI module is imported, and an instance of the FastAPI class is created.

-Request and Response Models: Two Pydantic models, ReviewItem and SentimentPrediction, are defined. ReviewItem represents the request body with a single field Review for the input text. SentimentPrediction represents the response body with a single field sentiment for the predicted sentiment.

-Model Loading: The pre-trained sentiment classification model (pickle file) is loaded into memory using the pickle.load() function and stored in the model variable.

-Prediction Endpoint: An API endpoint is defined using the @app.post decorator. The endpoint is accessible via the /predict route and accepts POST requests.

-Prediction Function: Inside the prediction endpoint, the input review text is tokenized using the pre-trained tokenizer. The tokenized input is then passed through the loaded model to obtain the predicted sentiment.

-Response: The predicted sentiment is wrapped in a SentimentPrediction object and returned as the API response.



## How to run

To use the API, send a POST request to the /predict route with a JSON payload containing the review text. The API will return a JSON response with the predicted sentiment.

Please note that you need to update the model path (model_path) in the API code to the correct file path of your trained model.



## Dependencies

Fastapi  
Pedantic  
Torch  
Transformers  
Pickle  



```python

```

