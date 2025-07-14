import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv('car data.csv')
print("Available columns:", df.columns.tolist())
df.rename(columns=lambda x: x.strip().replace(" ", "_"), inplace=True)
df = df[['name', 'fuel', 'km_driven', 'selling_price']]
df.dropna(inplace=True)

le_car = LabelEncoder()
le_fuel = LabelEncoder()
df['name'] = le_car.fit_transform(df['name'].str.lower())
df['fuel'] = le_fuel.fit_transform(df['fuel'].str.lower())

X = df[['name', 'fuel', 'km_driven']]
y = df['selling_price']

model = RandomForestRegressor()
model.fit(X, y)

joblib.dump(model, 'car_price_model.pkl')
joblib.dump(le_car, 'car_encoder.pkl')
joblib.dump(le_fuel, 'fuel_encoder.pkl')

print("\n=== Car Price Predictor ===")
car_name = input("Enter car name (e.g. Swift): ").lower()
fuel_type = input("Enter fuel type (petrol/diesel/cng): ").lower()
kms = int(input("Enter kilometers driven: "))

model = joblib.load('car_price_model.pkl')
le_car = joblib.load('car_encoder.pkl')
le_fuel = joblib.load('fuel_encoder.pkl')

if car_name not in le_car.classes_:
    print("Car name not found in training data.")
else:
    car_encoded = le_car.transform([car_name])[0]
    fuel_encoded = le_fuel.transform([fuel_type])[0]
    input_data = [[car_encoded, fuel_encoded, kms]]
    predicted_price = model.predict(input_data)[0]
    print(f"\nEstimated Selling Price: ₹{round(predicted_price, 2)} lakhs")
