from io import StringIO
import joblib
import numpy as np
import csv
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)

# Load the saved model and preprocessing steps
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
normalizer = joblib.load('normalizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    features = np.array(data)
    

    # Apply the same preprocessing steps
    features_standardized = scaler.transform(features.reshape(1, -1))
    features_normalized = normalizer.transform(features_standardized)

    # Make the prediction
    prediction = model.predict(features_normalized)

    
    return render_template('index.html', prediction_text='Prediction: {}'.format(prediction[0]),input_text='input_text: {}'.format(data))


if __name__ == '__main__':
    app.run(debug=True)
