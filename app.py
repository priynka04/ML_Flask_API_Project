from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('model.pkl')
target_names = ['setosa', 'versicolor', 'virginica']

@app.route('/')
def home():
    return "Iris ML Model API is running."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    # Basic validation
    if not data or 'features' not in data:
        return jsonify({'error': 'No input data or "features" key missing'}), 400
    
    features = data['features']
    if not isinstance(features, list) or len(features) != 4:
        return jsonify({'error': 'Features must be a list of 4 numeric values'}), 400
    
    try:
        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0]
        predicted_class = target_names[prediction]
        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
