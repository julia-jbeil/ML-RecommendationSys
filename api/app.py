from flask import Flask, render_template, request
import pandas as pd
import os
from restaurant_recommender import recommend_restaurants
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__, template_folder="../templates", static_folder="../static")
app.wsgi_app = ProxyFix(app.wsgi_app)

# ---------- Load data ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'restaurants.csv')
restaurants = pd.read_csv("https://github.com/julia-jbeil/ML-RecommendationSys/blob/main/data/restaurants.csv")

# ---------- Prepare filters ----------
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
