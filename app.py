
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("Used Laptop Price Predictor")

# Load the trained model and encoders
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

# Define the order of columns as used during training
# This assumes X was created using df.drop('Price_INR', axis=1) after encoding
# and that the original order of columns in df (before encoding) was as in df.head()
# and categorical columns were encoded in the order they appear in `categorical_cols`

# Based on the notebook, the order of columns in X is:
# 'Brand', 'Processor', 'RAM_GB', 'Storage_Type', 'Storage_GB', 'Screen_Size', 'Age_Years', 'Condition'

# Input widgets for user features
st.sidebar.header("Laptop Features")

# Categorical features
brand = st.sidebar.selectbox(
    "Brand",
    label_encoders['Brand'].classes_
)
processor = st.sidebar.selectbox(
    "Processor",
    label_encoders['Processor'].classes_
)
storage_type = st.sidebar.selectbox(
    "Storage Type",
    label_encoders['Storage_Type'].classes_
)
condition = st.sidebar.selectbox(
    "Condition",
    label_encoders['Condition'].classes_
)

# Numerical features
ram_gb = st.sidebar.number_input(
    "RAM (GB)",
    min_value=4, max_value=32, value=16, step=4
)
storage_gb = st.sidebar.number_input(
    "Storage (GB)",
    min_value=256, max_value=1024, value=512, step=256
)
screen_size = st.sidebar.number_input(
    "Screen Size (inches)",
    min_value=13.0, max_value=17.0, value=15.6, step=0.1
)
age_years = st.sidebar.number_input(
    "Age (Years)",
    min_value=1, max_value=7, value=3, step=1
)

# Prediction button
if st.button("Predict Price"):
    # Create a DataFrame from user input
    input_data = pd.DataFrame([{
        'Brand': brand,
        'Processor': processor,
        'RAM_GB': ram_gb,
        'Storage_Type': storage_type,
        'Storage_GB': storage_gb,
        'Screen_Size': screen_size,
        'Age_Years': age_years,
        'Condition': condition
    }])

    # Preprocess categorical features using loaded LabelEncoders
    for col, encoder in label_encoders.items():
        # Ensure the input data has the column before trying to transform
        if col in input_data.columns:
            # Apply transform, but ensure it handles unseen labels if necessary (though for selectbox, this shouldn't be an issue)
            input_data[col] = encoder.transform(input_data[col])

    # Make prediction
    prediction = model.predict(input_data)

    # Display result
    st.success(f"Predicted Laptop Price: ₹ {prediction[0]:,.2f}")
