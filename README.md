This is my first end-to-end machine learning project where I built a used car price prediction app. The goal was to estimate a car’s resale value based on inputs like brand, year, fuel type, seller type, ownership, and transmission.
I used a dataset from Kaggle, trained a regression model with scikit-learn, and built a simple UI using Streamlit to make predictions.


🔍 What the App Does
Takes input like car brand, year, fuel type, etc.
Predicts the approximate selling price using a trained model
Encodes categorical features using pre-fitted encoders
Provides an interactive UI built with Streamlit


🛠 Tools I Used
Python
Pandas, NumPy
Scikit-learn
Streamlit
Joblib
🚀 How to Run Locally

# Clone the repo or download the files
# Install required libraries
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
