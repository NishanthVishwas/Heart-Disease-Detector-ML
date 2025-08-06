# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
from model import train_models

app = Flask(__name__)

# Check if models exist, if not train them
if not os.path.exists('logistic_regression.pkl'):
    feature_columns = train_models()
else:
    # Load the dataset just to get column names
    df = pd.read_csv('heart.csv')
    feature_columns = df.drop('target', axis=1).columns.tolist()

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load all models
models = {}
model_files = ['logistic_regression.pkl', 'random_forest.pkl', 'svm.pkl']
model_names = ['Logistic Regression', 'Random Forest', 'Support Vector Machine']

for name, file in zip(model_names, model_files):
    with open(file, 'rb') as f:
        models[name] = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    return render_template('predict.html', features=feature_columns)

@app.route('/process', methods=['POST'])
def process():
    # Get form data
    data = request.form.to_dict()
    model_choice = data.pop('model_choice')
    
    # Convert to numeric values
    for key in data:
        data[key] = float(data[key])
    
    # Create a DataFrame with the input data
    input_df = pd.DataFrame([data])
    
    # Ensure columns are in the same order as during training
    input_df = input_df[feature_columns]
    
    # Scale the input features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction using the selected model
    model = models[model_choice]
    prediction = model.predict(input_scaled)[0]
    
    # Get probability if the model supports it
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(input_scaled)[0][1]
    else:
        # For models that don't support predict_proba (like some SVM configurations)
        probability = 0.5 if prediction == 1 else 0.5
    
    result = {
        'prediction': int(prediction),
        'probability': float(probability),
        'model_used': model_choice,
        'input_data': data
    }
    
    return render_template('results.html', result=result)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    # Get JSON data
    data = request.json
    model_choice = data.pop('model_choice')
    
    # Create a DataFrame with the input data
    input_df = pd.DataFrame([data])
    
    # Ensure columns are in the same order as during training
    input_df = input_df[feature_columns]
    
    # Scale the input features
    input_scaled = scaler.transform(input_df)
    
    # Make prediction using the selected model
    model = models[model_choice]
    prediction = model.predict(input_scaled)[0]
    
    # Get probability if the model supports it
    if hasattr(model, 'predict_proba'):
        probability = model.predict_proba(input_scaled)[0][1]
    else:
        probability = 0.5 if prediction == 1 else 0.5
    
    result = {
        'prediction': int(prediction),
        'probability': float(probability),
        'model_used': model_choice
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)