# =========================
# restaurant_recommender.py
# Advanced Hybrid Neural Recommender
# =========================

import os
import pickle
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# =========================
# 1️⃣ إعداد المسارات
# =========================
DATA_DIR = "data"
MODEL_DIR = "model"

os.makedirs(MODEL_DIR, exist_ok=True)

RESTAURANTS_FILE = os.path.join(os.path.dirname(__file__), DATA_DIR, "restaurants.csv")

MODEL_FILE = os.path.join(os.path.dirname(__file__), MODEL_DIR, "hybrid_model.pth")
ENCODERS_FILE = os.path.join(os.path.dirname(__file__), MODEL_DIR, "encoders.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 2️⃣ النموذج العصبي
# =========================
class HybridScoringNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# =========================
# 3️⃣ تحميل البيانات
# =========================
restaurants = pd.read_csv(RESTAURANTS_FILE)



# =========================
# 5️⃣ تحويل السعر
# =========================
def price_to_number(p):
    if pd.isna(p) or p.strip() == "":
        return 0
    return p.count("$") * 15

restaurants["price"] = restaurants["priceInterval"].apply(price_to_number)

# =========================
# 6️⃣ Encoding المطابخ
# =========================
restaurants["type"] = restaurants["type"].apply(eval)
mlb = MultiLabelBinarizer()
cuisine_matrix = mlb.fit_transform(restaurants["type"])
cuisine_df = pd.DataFrame(cuisine_matrix, columns=mlb.classes_)
restaurants = pd.concat([restaurants, cuisine_df], axis=1)

# =========================
# 7️⃣ Encoding المدينة + Scaling
# =========================
city_encoder = LabelEncoder()
restaurants["city_enc"] = city_encoder.fit_transform(restaurants["city"])
feature_cols = ["city_enc", "price"] + list(mlb.classes_)
X = restaurants[feature_cols].values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================
# 8️⃣ تدريب النموذج أو تحميله
# =========================
print("✅ Loading existing model...")
model = HybridScoringNet(X.shape[1]).to(device)
    # ✅ تحميل الأوزان فقط
state_dict = torch.load(MODEL_FILE, map_location=device)
model.load_state_dict(state_dict)

with open(ENCODERS_FILE, "rb") as f:
    scaler, city_encoder, mlb, feature_cols, restaurants = pickle.load(f)

model.eval()
# =========================
# 9️⃣ دالة التوصية
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
# 🔟 مثال تشغيل مباشر
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
