from tensorflow import keras
from tensorflow.keras import backend
import numpy as np
import json
import os
from flask import Flask, request
from flask import Flask, jsonify
import json

# create the Flask app
app = Flask(__name__)

@app.route('/coffee-points', methods=['POST'])
def json_example():
    # Get post data
    data = request.json
    request_data = json.loads(data)
    # Convert post data into an array for the dnn model
    X = np.array(request_data['Values'])
    # Predict results based on the model
    prediction = model.predict(X)
    return jsonify(prediction.tolist())


if __name__ == '__main__':
    # Load Coffee recommender DNN model
    directory = os.getcwd()
    model = keras.models.load_model(f"{directory}/models")
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)
