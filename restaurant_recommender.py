# =========================
# restaurant_recommender.py
# Advanced Hybrid Neural Recommender
# =========================

import os
import pickle
import pandas as pd
import torch
import warnings
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelEncoder
from sklearn.exceptions import InconsistentVersionWarning

# -------------------------
# Ignore scikit-learn version warnings when unpickling
# -------------------------
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# =========================
# 1️⃣ Paths
# =========================
MODEL_DIR = "model"
ENCODERS_FILE = os.path.join(os.path.dirname(__file__), MODEL_DIR, "encoders.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2️⃣ Load trained model and encoders
# =========================
with open(ENCODERS_FILE, "rb") as f:
    model, scaler, city_encoder, mlb, feature_cols, restaurants = pickle.load(f)

model.eval()

# =========================
# 3️⃣ Recommendation function
# =========================
def recommend_restaurants(city, cuisines=None, min_price=0, max_price=1000, top_n=5):
    if cuisines is None:
        cuisines = []

    try:
        city_code = city_encoder.transform([city])[0]
    except ValueError:
        return []

    mask = (
        (restaurants["city_enc"] == city_code) &
        (restaurants["price"] >= min_price) &
        (restaurants["price"] <= max_price)
    )

    valid_cuisines = [c for c in cuisines if c in mlb.classes_]
    if valid_cuisines:
        mask &= restaurants[valid_cuisines].sum(axis=1) > 0

    candidates = restaurants[mask].copy()
    if candidates.empty:
        return []

    X_pred = scaler.transform(candidates[feature_cols].values)
    X_pred_t = torch.tensor(X_pred, dtype=torch.float32).to(device)

    with torch.no_grad():
        scores = model(X_pred_t).cpu().numpy().flatten()

    candidates["score"] = scores
    return candidates.sort_values("score", ascending=False).head(top_n)[
        ["itemId", "name", "city", "priceInterval", "price", "avg_rating", "type", "score", "url"]
    ].to_dict(orient="records")

# =========================
# 🔟 Direct run example
# =========================
if __name__ == "__main__":
    recs = recommend_restaurants(
        city="Gijon",
        cuisines=["Spanish", "Seafood"],
        min_price=10,
        max_price=40,
        top_n=5
    )
    for r in recs:
        print(r)
س