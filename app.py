from flask import Flask, render_template, request
import pandas as pd
from restaurant_recommender import recommend_restaurants

app = Flask(__name__)

# Load restaurants
restaurants = pd.read_csv('data/restaurants.csv')

# Unique cities & cuisines
cities = sorted(restaurants['city'].dropna().unique())
cuisine_list = set()

# Extract cuisines from dataset (restaurant["type"] column contains list-like)
for row in restaurants['type'].dropna():
    if isinstance(row, str):
        # Remove special chars and convert "[ 'A', 'B' ]" → ["A", "B"]
        cleaned = row.strip("[]").replace("'", "")
        parts = [c.strip() for c in cleaned.split(",") if c.strip()]
        cuisine_list.update(parts)

cuisines = sorted(list(cuisine_list))

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        city = request.form.get('city')
        cuisines_selected = request.form.getlist('cuisines')
        min_price = int(request.form.get('min_price', 0))
        max_price = int(request.form.get('max_price', 1000))
        top_n = int(request.form.get('top_n', 5))

        recommendations = recommend_restaurants(
            city=city.strip(),
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

if __name__ == "__main__":
    app.run(debug=True)
