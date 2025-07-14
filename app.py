import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open('updated_model.pkl', 'rb'))
car_encoder = pickle.load(open('car_encoder.pkl', 'rb'))
fuel_encoder = pickle.load(open('fuel_encoder.pkl', 'rb'))
owner_encoder = pickle.load(open('owner_encoder.pkl', 'rb'))

df = pd.read_csv("car data.csv")
car_fuel_map = df.groupby("name")["fuel"].unique().apply(list).to_dict()
owner_list = sorted(owner_encoder.classes_)

st.title("🚗 Car Price Predictor")

car_list = sorted(car_encoder.classes_)
selected_car = st.selectbox("Select Car", car_list)

available_fuels = car_fuel_map.get(selected_car, fuel_encoder.classes_)
fuel = st.radio("Select Fuel Type", sorted(available_fuels))

year = st.number_input("Year of Purchase", min_value=1995, max_value=2025, value=2015)
owner = st.selectbox("Ownership Type", owner_list)
kms = st.number_input("Kilometers Driven", min_value=0, step=100)

if st.button("Predict Price"):
    try:
        car_encoded = car_encoder.transform([selected_car])[0]
        fuel_encoded = fuel_encoder.transform([fuel])[0]
        owner_encoded = owner_encoder.transform([owner])[0]
        input_data = np.array([[car_encoded, fuel_encoded, kms, year, owner_encoded]])
        predicted_price = model.predict(input_data)[0]
        st.success(f"Estimated Price: ₹{predicted_price:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
