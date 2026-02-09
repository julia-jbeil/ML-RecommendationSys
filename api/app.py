from flask import Flask, render_template, request
import pandas as pd
import os
import sys

# ---------------- Fix import from parent folder ----------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)  # Add project root to Python path

import restaurant_recommender  # ✅ Import the module

# Optional: import recommend_restaurants function directly
recommend_restaurants = restaurant_recommender.recommend_restaurants

from werkzeug.middleware.proxy_fix import ProxyFix

# ---------------- Flask app setup ----------------
app = Flask(
    __name__,
    template_folder=os.path.join(PROJECT_ROOT, "templates"),
    static_folder=os.path.join(PROJECT_ROOT, "static")
)
app.wsgi_app = ProxyFix(app.wsgi_app)

# ---------------- Load data ----------------
restaurants = pd.read_csv(
    "https://raw.githubusercontent.com/julia-jbeil/ML-RecommendationSys/main/data/restaurants.csv"
)

# ---------------- Prepare filters ----------------
cities = sorted(restaurants['city'].dropna().unique())
cuisine_list = set()
for row in restaurants['type'].dropna():
    if isinstance(row, str):
        cleaned = row.strip("[]").replace("'", "")
        parts = [c.strip() for c in cleaned.split(",") if c.strip()]
        cuisine_list.update(parts)
cuisines = sorted(cuisine_list)

# ---------- Routes ----------
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []

    if request.method == 'POST':
        city = (request.form.get('city') or "").strip()
        cuisines_selected = request.form.getlist('cuisines')
        min_price = int(request.form.get('min_price') or 0)
        max_price = int(request.form.get('max_price') or 1000)
        top_n = 5

        recommendations = recommend_restaurants(
            city=city,
            cuisines=cuisines_selected,
            min_price=min_price,
            max_price=max_price,
            top_n=top_n
        )

    return render_template(
        'index.html',
        cities=cities,
        cuisines=cuisines,
        recommendations=recommendations
    )

# ---------- Vercel serverless entry ----------
def handler(environ, start_response):
    return app(environ, start_response)
