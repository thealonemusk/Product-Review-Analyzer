import numpy as np
from flask import Flask, request, jsonify
from flasgger import Swagger
import pickle

# Create Flask app
flask_app = Flask(__name__)
Swagger(flask_app)
model = pickle.load(open("test.pkl", "rb"))

@flask_app.route("/predict", methods=["POST","GET"])
def predict():
    if(request.method=="GET"):
        return jsonify("Hello World")
    
    """
    Predicts the flower species based on input features.
    ---
    parameters:
      - name: features
        in: body
        required: true
        type: array
        items:
          type: number
        minItems: 4
        maxItems: 4
    responses:
      200:
        description: Prediction result
        schema:
          properties:
            prediction:
              type: string
    """
    float_features = request.json["features"]
    features = [np.array(float_features)]
    prediction = model.predict(features)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    flask_app.run(debug=True)
