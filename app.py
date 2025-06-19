from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Allow frontend to access backend

# Load model dan scaler
try:
    model = load('model_klasifikasi.joblib')
    scaler = load('scaler.joblib')
    print("Model dan scaler berhasil dimuat")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Heart Disease Classification API',
        'status': 'running',
        'endpoints': ['/predict']
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded properly'}), 500
        
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400
            
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        
        return jsonify({
            'prediction': int(prediction[0]),
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)