import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("car data.csv")

# Clean important columns
df = df[["name", "fuel", "owner", "km_driven", "year", "selling_price"]].dropna()

df["km_driven"] = pd.to_numeric(df["km_driven"], errors="coerce")
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")
df.dropna(inplace=True)

# Encode labels
car_enc = LabelEncoder()
fuel_enc = LabelEncoder()
owner_enc = LabelEncoder()

df["car_enc"] = car_enc.fit_transform(df["name"])
df["fuel_enc"] = fuel_enc.fit_transform(df["fuel"])
df["owner_enc"] = owner_enc.fit_transform(df["owner"])

# Log transform for KM
df["kms_log"] = np.log1p(df["km_driven"])

# Final training features (5 features ONLY)
X = df[["car_enc", "fuel_enc", "kms_log", "year", "owner_enc"]]
y = df["selling_price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# Train Random Forest
model = RandomForestRegressor(
    n_estimators=300, max_depth=20, random_state=42
)
model.fit(X_train, y_train)

# Save model + encoders
joblib.dump(model, "updated_model.pkl")
joblib.dump(car_enc, "car_encoder.pkl")
joblib.dump(fuel_enc, "fuel_encoder.pkl")
joblib.dump(owner_enc, "owner_encoder.pkl")

print("Training complete. Files saved successfully!")
