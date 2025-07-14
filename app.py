import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("🚗 Car Price Predictor")

st.markdown("---")

try:
    model = joblib.load('car_price_model.pkl')
    le_car = joblib.load('car_encoder.pkl')
    le_fuel = joblib.load('fuel_encoder.pkl')
except Exception as e:
    st.error("Error loading model or encoders. Please check your files.")
    st.stop()

car_names = sorted(le_car.classes_)
fuel_types = ["petrol", "diesel", "cng"]

st.markdown("### 📝 Enter Car Details Below")

with st.form("car_price_form"):
    car_name = st.selectbox("Select Car Name", car_names)
    fuel_type = st.radio("Select Fuel Type", fuel_types, horizontal=True)
    kms = st.number_input("Kilometers Driven", min_value=0, value=1000, step=500)

    submitted = st.form_submit_button("Predict Price")

if submitted:
    try:
        car_encoded = le_car.transform([car_name])[0]
        fuel_encoded = le_fuel.transform([fuel_type])[0]
        input_data = np.array([[car_encoded, fuel_encoded, kms]])
        predicted_price = model.predict(input_data)[0]
        price_lakhs = round(predicted_price / 100000, 2)

        st.success(f"💰 Estimated Selling Price: ₹{price_lakhs} lakhs")
        st.balloons()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
