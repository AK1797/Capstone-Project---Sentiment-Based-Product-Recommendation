from flask import Flask, render_template, request
import pandas as pd
from model import get_final_recommendations
import joblib

app = Flask(__name__)

# Load precomputed product stats (small) for template rendering if needed
# product_stats_ext.pkl must be in same folder
product_stats = pd.read_pickle('product_stats_ext.pkl')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    username = request.form.get('username', '').strip()
    if username == '':
        return render_template('index.html', error="Please enter a username.")
    try:
        results = get_final_recommendations(username, top_k_candidates=20, final_k=5)
        return render_template('index.html', username=username, recommendations=results)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
