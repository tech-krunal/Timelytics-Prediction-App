import streamlit as st
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import time

# Set Streamlit page configuration
st.set_page_config(
    page_title="Timelytics - Delivery Time Predictor",
    page_icon=":package:",
    layout="wide"
)

# Load the trained ensemble model
modelfile = "./voting_model.pkl"
voting_model = pickle.load(open(modelfile, "rb"))

# Function to predict delivery time
def predict_delivery_time(purchase_dow, purchase_month, year, product_size_cm3,
                          product_weight_g, geolocation_state_customer,
                          geolocation_state_seller, distance):
    prediction = voting_model.predict(np.array([
        [purchase_dow, purchase_month, year, product_size_cm3,
         product_weight_g, geolocation_state_customer,
         geolocation_state_seller, distance]
    ]))
    return round(prediction[0])

# Sidebar for input parameters
with st.sidebar:
    img = Image.open("./assets/supply_chain_optimisation.jpg")
    st.image(img)
    st.header("Input Parameters")
    
    purchase_dow = st.selectbox("Purchased Day of the Week", options=list(range(7)), format_func=lambda x: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"][x])
    purchase_month = st.selectbox("Purchased Month", options=list(range(1, 13)))
    year = st.slider("Purchased Year", 2017, 2025, 2018)
    product_size_cm3 = st.number_input("Product Size in cm³", min_value=1, value=9328)
    product_weight_g = st.number_input("Product Weight in grams", min_value=1, value=1800)
    geolocation_state_customer = st.number_input("Customer State Code", min_value=1, value=10)
    geolocation_state_seller = st.number_input("Seller State Code", min_value=1, value=20)
    distance = st.number_input("Distance (in km)", min_value=0.1, value=475.35)
    
    submit = st.button("Predict Delivery Time")

# Main content
st.title("Timelytics :rocket:")
st.subheader("Optimizing Supply Chain with AI-Powered Delivery Time Predictions")

with st.container():
    st.header("Prediction Result")
    if submit:
        with st.spinner("Calculating estimated delivery time..."):
            time.sleep(2)  # Simulating loading time
            prediction = predict_delivery_time(
                purchase_dow, purchase_month, year, product_size_cm3,
                product_weight_g, geolocation_state_customer,
                geolocation_state_seller, distance
            )
            st.success(f"Estimated delivery time: **{prediction} days**")

# Sample dataset
sample_data = {
    "Purchased Day of the Week": [0, 3, 1],
    "Purchased Month": [6, 3, 1],
    "Purchased Year": [2018, 2017, 2018],
    "Product Size in cm³": [37206, 63714, 54816],
    "Product Weight in grams": [16250, 7249, 9600],
    "Customer State Code": [25, 25, 25],
    "Seller State Code": [20, 7, 20],
    "Distance (in km)": [247.94, 250.35, 4.915]
}

st.header("Sample Dataset")
st.dataframe(pd.DataFrame(sample_data))
