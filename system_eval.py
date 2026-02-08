# =========================
# evaluate_model.py
# =========================

import os
import pickle
import torch
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from restaurant_recommender import HybridScoringNet, DATA_DIR, MODEL_DIR

# =========================
# 1️⃣ Paths setup
# =========================
RESTAURANTS_FILE = os.path.join(os.path.dirname(__file__), DATA_DIR, "restaurants.csv")
MODEL_FILE = os.path.join(os.path.dirname(__file__), MODEL_DIR, "hybrid_model.pth")
ENCODERS_FILE = os.path.join(os.path.dirname(__file__), MODEL_DIR, "encoders.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2️⃣ Load data and model
# =========================
restaurants = pd.read_csv(RESTAURANTS_FILE)

with open(ENCODERS_FILE, "rb") as f:
    scaler, city_encoder, mlb, feature_cols, restaurants = pickle.load(f)

X = restaurants[feature_cols].values
y_true = restaurants["avg_rating"].values / 50.0  # same scaling as training

X_scaled = scaler.transform(X)
X_t = torch.tensor(X_scaled, dtype=torch.float32).to(device)

model = HybridScoringNet(X.shape[1]).to(device)
state_dict = torch.load(MODEL_FILE, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# =========================
# 3️⃣ Make predictions
# =========================
with torch.no_grad():
    y_pred = model(X_t).cpu().numpy().flatten()

# =========================
# 4️⃣ Evaluate model
# =========================
mse = mean_squared_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
import numpy as np
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print(f"📊 Model Evaluation:")
print(f"MSE  = {mse:.4f}")
print(f"RMSE = {rmse:.4f}")
print(f"R²   = {r2:.4f}")

# =========================
# 5️⃣ Plot results
# =========================
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # ideal line y=x
plt.xlabel("True Ratings (Normalized)")
plt.ylabel("Predicted Ratings (Normalized)")
plt.title("True vs Predicted Ratings")
plt.grid(True)
plt.show()
