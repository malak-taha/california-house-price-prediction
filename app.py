import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. Load the trained model
with open('xgboost_housing_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# 2. App interface styling
st.title("🏡 California House Price Prediction App")
st.write("Adjust the sliders below to predict the house price dynamically.")

st.header("House Features")

# 3. Sliders input aligned with California features layout
med_inc = st.slider("Median Income (in tens of thousands)", 0.5, 15.0, 3.5)
house_age = st.slider("House Age", 1.0, 52.0, 28.0)
ave_rooms = st.slider("Average Rooms", 1.0, 10.0, 5.0)
ave_bedrms = st.slider("Average Bedrooms", 1.0, 5.0, 1.0)
population = st.slider("Population in the block", 3.0, 35000.0, 1400.0)
ave_occup = st.slider("Average House Occupancy", 1.0, 10.0, 3.0)
latitude = st.slider("Latitude", 32.5, 42.0, 35.5)
longitude = st.slider("Longitude", -124.3, -114.3, -119.5)

# 4. Trigger Prediction
if st.button("Predict Price"):
    input_data = np.array([[med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]])
    prediction = loaded_model.predict(input_data)
    
    # Scale result back to dollars (as California target is in $100k units)
    final_price = prediction[0] * 100000
    st.success(f"💰 Estimated House Price: ${final_price:,.2f}")
