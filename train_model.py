import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("car data.csv")
df = df[df['selling_price'] > 50000]
df = df[df['km_driven'] < 500000]

car_encoder = LabelEncoder()
fuel_encoder = LabelEncoder()
owner_encoder = LabelEncoder()

df['name'] = car_encoder.fit_transform(df['name'])
df['fuel'] = fuel_encoder.fit_transform(df['fuel'])
df['owner'] = owner_encoder.fit_transform(df['owner'])

X = df[['name', 'fuel', 'km_driven', 'year', 'owner']]
y = df['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open("updated_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("car_encoder.pkl", "wb") as f:
    pickle.dump(car_encoder, f)

with open("fuel_encoder.pkl", "wb") as f:
    pickle.dump(fuel_encoder, f)

with open("owner_encoder.pkl", "wb") as f:
    pickle.dump(owner_encoder, f)

print("✅ Model and encoders saved successfully.")
