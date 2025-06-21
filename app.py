from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load('model.pkl')

# Define class names (Iris dataset target names)
target_names = ['setosa', 'versicolor', 'virginica']

@app.route('/')
def home():
    return "Iris ML Model API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    prediction = model.predict(features)
    predicted_class = target_names[int(prediction[0])]   # Use target_names here
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
