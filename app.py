##  Import Required Libraries
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from flask import Flask, render_template, request, jsonify


## Load Pre-Trained Model & Preprocessing Objects 
model = tf.keras.models.load_model('models/model.h5')

with open('preprocessors/label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('preprocessors/on_hot_encoder_geography.pkl', 'rb') as file:
    on_hot_encoder_geography = pickle.load(file)

with open('preprocessors/scalar.pkl', 'rb') as file:
    scaler = pickle.load(file)    

# Initialize Flask App
app = Flask(__name__)

## Define Home Route
@app.route("/")
def home():
    return render_template('index.html')

## Define Prediction Endpoint
@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    
    # Convert categorical inputs
    gender = label_encoder_gender.transform([data['gender']])[0]
    geography = on_hot_encoder_geography.transform([[data['geography']]])
    geo_encoded_df = pd.DataFrame(geography, columns=on_hot_encoder_geography.get_feature_names_out())
    
    input_data = pd.DataFrame({
        "CreditScore": [data['credit_score']],
        "Gender": [gender],
        "Age": [data['age']],
        "Tenure": [data['tenure']],
        "Balance": [data['balance']],
        "NumOfProducts": [data['num_of_products']],
        "HasCrCard": [1 if data['has_credit_card'] == "Yes" else 0],
        "IsActiveMember": [1 if data['is_active_member'] == "Yes" else 0],
        "EstimatedSalary": [data['estimated_salary']]
    })
    
    input_data = pd.concat([input_data, geo_encoded_df], axis=1)
    input_scaled = scaler.transform(input_data)
    
    prediction = model.predict(input_scaled)[0][0]
    
    return jsonify({
            'input_data': data,
            'churn_probability': round(float(prediction), 2)
        })
# Run the Flask App
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
