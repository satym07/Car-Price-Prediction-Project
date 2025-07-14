import streamlit as st
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))
car_encoder = pickle.load(open('car_encoder.pkl', 'rb'))
fuel_encoder = pickle.load(open('fuel_encoder.pkl', 'rb'))

st.title("🚗 Car Price Predictor")

car_list = sorted(car_encoder.classes_)
car = st.selectbox("Select Car", car_list)
fuel = st.radio("Select Fuel Type", fuel_encoder.classes_)
kms = st.number_input("Kilometers Driven", min_value=0, step=100)

if st.button("Predict Price"):
    try:
        car_encoded = car_encoder.transform([car])[0]
        fuel_encoded = fuel_encoder.transform([fuel])[0]
        input_data = np.array([[car_encoded, fuel_encoded, kms]])
        predicted_price = model.predict(input_data)[0]
        st.success(f"Estimated Price: ₹{predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
