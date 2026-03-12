# ───────────────────────────────────────────
# AgriTech — ML Model Builder
# Run this file ONCE to train and save models
# ───────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
import pickle
import os

print("🌾 AgriTech Model Training Started...")
print("=" * 45)

# ── 1. LOAD DATASET ──────────────────────────
df = pd.read_csv('data/data_season.csv')
df['Soil type'] = df['Soil type'].fillna('Unknown')
print(f"✅ Dataset loaded — {df.shape[0]} records, {df.shape[1]} columns")

# ── 2. ENCODE TEXT COLUMNS ───────────────────
le_location   = LabelEncoder()
le_soil       = LabelEncoder()
le_irrigation = LabelEncoder()
le_crop       = LabelEncoder()
le_season     = LabelEncoder()
le_yield_cat  = LabelEncoder()

df['Location_enc']   = le_location.fit_transform(df['Location'])
df['Soil_enc']       = le_soil.fit_transform(df['Soil type'])
df['Irrigation_enc'] = le_irrigation.fit_transform(df['Irrigation'])
df['Crop_enc']       = le_crop.fit_transform(df['Crops'])
df['Season_enc']     = le_season.fit_transform(df['Season'])

print("✅ Data encoding done")

# ── 3. DEFINE FEATURES ───────────────────────
features_recommend = [
    'Year', 'Location_enc', 'Area', 'Rainfall',
    'Temperature', 'Soil_enc', 'Irrigation_enc',
    'Humidity', 'Season_enc'
]

features_price = features_recommend + ['Crop_enc']

# ── 4. MODEL A — CROP RECOMMENDATION ─────────
print("\n📌 Training Crop Recommendation Model...")
X = df[features_recommend]
y = df['Crop_enc']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

crop_model = RandomForestClassifier(n_estimators=200, random_state=42)
crop_model.fit(X_train, y_train)
acc = accuracy_score(y_test, crop_model.predict(X_test))
print(f"✅ Crop Recommendation Accuracy: {acc*100:.1f}%")

# ── 5. MODEL B — YIELD PREDICTION ────────────
print("\n📌 Training Yield Prediction Model...")

# Convert yield to Low / Medium / High category
df['yield_cat'] = pd.qcut(
    df['yeilds'], q=3, labels=['Low', 'Medium', 'High']
)
df['yield_cat_enc'] = le_yield_cat.fit_transform(df['yield_cat'])

X2 = df[features_price]
y2 = df['yield_cat_enc']

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

yield_model = RandomForestClassifier(n_estimators=200, random_state=42)
yield_model.fit(X_train2, y_train2)
acc2 = accuracy_score(y_test2, yield_model.predict(X_test2))
print(f"✅ Yield Prediction Accuracy: {acc2*100:.1f}%")

# ── 6. MODEL C — PRICE PREDICTION ────────────
print("\n📌 Training Price Prediction Model...")

df['price_log'] = np.log1p(df['price'])
X3 = df[features_price]
y3 = df['price_log']

X_train3, X_test3, y_train3, y_test3 = train_test_split(
    X3, y3, test_size=0.2, random_state=42
)

price_model = RandomForestRegressor(n_estimators=200, random_state=42)
price_model.fit(X_train3, y_train3)
r2 = r2_score(y_test3, price_model.predict(X_test3))
print(f"✅ Price Prediction R² Score: {r2*100:.1f}%")

# ── 7. SAVE ALL MODELS ────────────────────────
print("\n💾 Saving all models...")

os.makedirs('models', exist_ok=True)

pickle.dump(crop_model,   open('models/crop_model.pkl',   'wb'))
pickle.dump(yield_model,  open('models/yield_model.pkl',  'wb'))
pickle.dump(price_model,  open('models/price_model.pkl',  'wb'))

# Save encoders too — needed later in Flask
pickle.dump(le_location,   open('models/le_location.pkl',   'wb'))
pickle.dump(le_soil,       open('models/le_soil.pkl',       'wb'))
pickle.dump(le_irrigation, open('models/le_irrigation.pkl', 'wb'))
pickle.dump(le_crop,       open('models/le_crop.pkl',       'wb'))
pickle.dump(le_season,     open('models/le_season.pkl',     'wb'))
pickle.dump(le_yield_cat,  open('models/le_yield_cat.pkl',  'wb'))

print("✅ All models saved in /models folder!")

# ── 8. PRINT LABEL INFO ───────────────────────
print("\n📋 Label Reference (save this!):")
print(f"Crops     : {list(le_crop.classes_)}")
print(f"Locations : {list(le_location.classes_)}")
print(f"Seasons   : {list(le_season.classes_)}")
print(f"Soil Types: {list(le_soil.classes_)}")
print(f"Irrigation: {list(le_irrigation.classes_)}")
print(f"Yield Cats: {list(le_yield_cat.classes_)}")

print("\n" + "=" * 45)
print("🎉 All 3 models trained and saved successfully!")
print("Next step: Run app.py to start the web server")