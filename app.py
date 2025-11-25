import streamlit as st
import joblib 
import numpy as np
import pandas as pd
import base64


# background image

def background(img):
    with open(img, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-attachment: fixed;
            }}
            .block-container {{
                background: rgba(0,0,0,0.25);
                backdrop-filter: blur(6px);
                padding: 20px;
                border-radius: 10px;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

background("background.jpg")


# loading files

model = joblib.load("updated_model.pkl")
car_encoder = joblib.load("car_encoder.pkl")
fuel_encoder = joblib.load("fuel_encoder.pkl")
owner_encoder = joblib.load("owner_encoder.pkl")

data = pd.read_csv("car data.csv")
fuel_map = data.groupby("name")["fuel"].unique().apply(list).to_dict()

st.set_page_config(page_title="Car Price Predictor", layout="wide")

# tabs
tab1, tab2, tab3 = st.tabs(["Prediction", "Dataset", "Feature Importance"])

# prediction tab
with tab1:
    st.title("Car Price Prediction")

    with st.expander("Enter Details", expanded=True):
        car_list = sorted(car_encoder.classes_)
        car_name = st.selectbox("Car Model", car_list)

        fuels = fuel_map.get(car_name, fuel_encoder.classes_)
        fuel = st.radio("Fuel Type", sorted(fuels))

        c1, c2 = st.columns(2)
        with c1:
            year = st.number_input("Year of Purchase", 1995, 2025, 2015)
        with c2:
            owner = st.selectbox("Ownership Type", sorted(owner_encoder.classes_))

        kms = st.number_input("Kilometers Driven", 0, step=500)

    # prediction
    if st.button("Predict Price"):
        try:
            car_val = car_encoder.transform([car_name])[0]
            fuel_val = fuel_encoder.transform([fuel])[0]
            owner_val = owner_encoder.transform([owner])[0]
            km_log = np.log1p(kms)

            # FINAL input order matching the 5-feature model
            x = np.array([[car_val, fuel_val, km_log, year, owner_val]])

            result = model.predict(x)[0]

            st.success(f"Estimated Price: ₹{result:,.0f}")
        except Exception as e:
            st.error(f"Error: {e}")

# dataset tab
with tab2:
    st.title("Dataset Preview")
    st.dataframe(data.head(15))

# feature importance tab
with tab3:
    st.title("Feature Importance")

    try:
        importance = model.feature_importances_
        features = ["Car Name", "Fuel Type", "KMs (Log)", "Year", "Owner"]

        imp_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values("Importance", ascending=False)

        st.bar_chart(imp_df.set_index("Feature"))
        st.write("Higher value means more influence on the prediction.")
    except:
        st.write("Feature importance is not available for this model.")
