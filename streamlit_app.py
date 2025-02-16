import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import streamlit as st 


## Load model 
model = tf.keras.models.load_model('models/model.h5')

with open('preprocessors/label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('preprocessors/on_hot_encoder_geography.pkl', 'rb') as file:
    on_hot_encoder_geography = pickle.load(file)

with open('preprocessors/scalar.pkl', 'rb') as file:
    scaler = pickle.load(file)    

# Streamlit UI
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")
st.markdown("""
    <style>
    .big-font { font-size:24px !important; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to churn based on input features.")

# User Input 
# Sidebar for User Input
st.sidebar.header("🔹 Enter Customer Details")

gender = st.sidebar.selectbox("Gender", label_encoder_gender.classes_)
geography = st.sidebar.selectbox('Geography', on_hot_encoder_geography.categories_[0])
age = st.sidebar.slider("Age", 18, 92, 30)
balance = st.sidebar.number_input('Balance', min_value=0.0, step=100.0, format="%.2f")
credit_score = st.sidebar.slider('Credit Score', 300, 900, 650)
estimated_salary = st.sidebar.number_input('Estimated Salary', min_value=0.0, step=500.0, format="%.2f")
tenure = st.sidebar.slider('Tenure (Years)', 0, 10, 3)
num_of_prod = st.sidebar.slider('Number of Products', 1, 4, 1)
has_cr_card = st.sidebar.selectbox('Has Credit Card?', ['No', 'Yes'])
is_active_member = st.sidebar.selectbox('Is Active Member?', ['No', 'Yes'])

# Convert categorical inputs to numerical
has_cr_card = 1 if has_cr_card == "Yes" else 0
is_active_member = 1 if is_active_member == "Yes" else 0

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
data = pd.concat([input_data.drop(['Geography'], axis=1), geo_encoded_df], axis=1)

data_scaled = scaler.transform(data)

# Show Data Before Prediction
st.subheader("📋 Input Data Preview")
st.dataframe(input_data)

# Predict churn 
if st.button("🔍 Predict Churn"):
    prediction = model.predict(data_scaled)[0][0]
    
    # Display Prediction
    st.subheader("📌 Prediction Result")
    st.markdown(f"<p class='big-font'>Customer Churn Probability: <b>{prediction:.2%}</b></p>", unsafe_allow_html=True)
    
    # Progress Bar
    st.progress(float(prediction))

    if prediction > 0.5:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is not likely to churn.")