import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import streamlit as st 


## Load model 
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('on_hot_encoder_geography.pkl', 'rb') as file:
    on_hot_encoder_geography = pickle.load(file)

with open('scalar.pkl', 'rb') as file:
    scaler = pickle.load(file)    

# Streamlit APP
st.title("Customer Churn Prediction")

# User Input 

geography = st.selectbox('GeoGraphy', on_hot_encoder_geography.categories_[0])
gender = st.selectbox("Gender", label_encoder_gender.classes_)
age = st.slider("Age", 18, 92)
balance = st.number_input('Balance')
credit_score = st.slider('Credit Score', 300, 900)
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_prod = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data =   pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [label_encoder_gender.transform([gender])[0]],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_prod],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
  })


geo_encoded  = on_hot_encoder_geography.transform([input_data['Geography']])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=on_hot_encoder_geography.get_feature_names_out())
input_data = pd.concat([input_data.drop(['Geography'], axis=1), geo_encoded_df], axis=1)

st.write(input_data)

input_data_scaled = scaler.transform(input_data)
# Predict churn 

prediction = model.predict(input_data_scaled)[0][0]
st.write(f"Customer Churn Probablity : {prediction}")
if prediction > 0.5:
    st.write("Customer is likely to churn.")
else:
    st.write("Customer is not likely to churn.")