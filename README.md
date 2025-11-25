🚗 Car Price Prediction — End-to-End Machine Learning Project
This project is an end-to-end Used Car Price Prediction system that estimates the resale value of a car based on its specifications. It includes a complete ML workflow along with a polished, interactive Streamlit web interface featuring a blurred automotive-themed background image.
🧠 Project Overview
The goal of this project is to predict a car’s selling price using details such as:
Car Model
Fuel Type
Year of Purchase
Ownership Type
Kilometers Driven
The model is trained on a used-car dataset and integrated into a modern UI for smooth user interaction.
🎨 Application UI
The app contains:
A modern blurred background for a premium feel
A sidebar-style tab navigation (Prediction • Dataset • Feature Importance)
Dropdowns, radio buttons, and numeric inputs
A clear “Predict Price” button
Neatly aligned input sections (“Enter Details” panel)
The design is clean, dark-themed, and user-friendly.
🔍 Features of the App
✔ Enter car details using interactive widgets
✔ Encodes categorical features automatically
✔ Predicts used car price in real-time
✔ Tabs for Dataset preview & Feature Importance
✔ Background image for improved UI aesthetics
🛠 Technologies Used
Python
Pandas, NumPy
Scikit-learn
Streamlit
Joblib
📁 Project Structure
├── app.py               # Streamlit user interface
├── train_model.py       # Model training pipeline
├── car data.csv         # Training dataset
├── background.jpg       # Background image used in UI
└── README.md            # Documentation
🚀 How to Run the Project
1️⃣ Clone the Repository
git clone <your-repo-url>
cd <your-repo-folder>
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the App
streamlit run app.py
The UI will open automatically in the browser.
📊 Model Information
The model predicts car price using regression algorithms trained on historical used car data.
Evaluation metrics include:
MAE
RMSE
R² Score
You can add your specific values (optional).
⭐ Summary
This project demonstrates:
End-to-end ML development
Model training & evaluation
Feature encoding
Clean UI development with Streamlit
Practical problem-solving using data
