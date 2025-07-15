This is my first end-to-end machine learning project where I built a used car price prediction app. The goal was to estimate a car’s resale value based on inputs like brand, year, fuel type, seller type, ownership, and transmission.
I used a dataset from Kaggle, trained a regression model with scikit-learn, and built a simple UI using Streamlit to make predictions.


🔍 What the App Does
Takes input like car brand, year, fuel type, etc.
Predicts the approximate selling price using a trained model
Encodes categorical features using pre-fitted encoders
Provides an interactive UI built with Streamlit


📁 Project Structure
├── car data.csv             # Dataset
├── train_model.py           # Model training and encoder saving
├── app.py                   # Streamlit app
├── updated_model.pkl        # Trained model
├── car_encoder.pkl          # Car name encoder
├── fuel_encoder.pkl         # Fuel type encoder
├── owner_encoder.pkl        # Owner type encoder
├── run_app.sh               # Shortcut to run the app


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
