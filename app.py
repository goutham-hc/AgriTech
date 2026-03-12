# ─────────────────────────────────────────────
# AgriTech — Flask Backend (app.py)
# This file connects your ML models to the web
# ─────────────────────────────────────────────

from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# ── LOAD ALL SAVED MODELS ─────────────────────
print("🔄 Loading models...")

crop_model    = pickle.load(open('models/crop_model.pkl',   'rb'))
yield_model   = pickle.load(open('models/yield_model.pkl',  'rb'))
price_model   = pickle.load(open('models/price_model.pkl',  'rb'))

le_location   = pickle.load(open('models/le_location.pkl',   'rb'))
le_soil       = pickle.load(open('models/le_soil.pkl',       'rb'))
le_irrigation = pickle.load(open('models/le_irrigation.pkl', 'rb'))
le_crop       = pickle.load(open('models/le_crop.pkl',       'rb'))
le_season     = pickle.load(open('models/le_season.pkl',     'rb'))
le_yield_cat  = pickle.load(open('models/le_yield_cat.pkl',  'rb'))

print("✅ All models loaded!")

# ── PAGE ROUTES ───────────────────────────────

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Crop Recommendation page
@app.route('/recommend')
def recommend():
    return render_template('recommend.html')

# Yield Prediction page
@app.route('/yield')
def yield_page():
    return render_template('yield.html')

# Price Prediction page
@app.route('/price')
def price():
    return render_template('price.html')

# ── API ROUTES ────────────────────────────────

# API 1 — Crop Recommendation
@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    try:
        data = request.json

        # Encode inputs
        loc  = le_location.transform([data['location']])[0]
        soil = le_soil.transform([data['soil']])[0]
        irr  = le_irrigation.transform([data['irrigation']])[0]
        seas = le_season.transform([data['season']])[0]

        features = [[
            int(data['year']),
            loc,
            float(data['area']),
            float(data['rainfall']),
            float(data['temperature']),
            soil,
            irr,
            float(data['humidity']),
            seas
        ]]

        prediction = crop_model.predict(features)[0]
        crop_name  = le_crop.inverse_transform([prediction])[0]

        # Get top 3 crop probabilities
        proba      = crop_model.predict_proba(features)[0]
        top3_idx   = proba.argsort()[-3:][::-1]
        top3_crops = [
            {
                'crop': le_crop.inverse_transform([i])[0],
                'confidence': round(proba[i] * 100, 1)
            }
            for i in top3_idx
        ]

        return jsonify({
            'success': True,
            'recommended_crop': crop_name,
            'top3': top3_crops
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# API 2 — Yield Prediction
@app.route('/api/yield', methods=['POST'])
def api_yield():
    try:
        data = request.json

        loc  = le_location.transform([data['location']])[0]
        soil = le_soil.transform([data['soil']])[0]
        irr  = le_irrigation.transform([data['irrigation']])[0]
        crop = le_crop.transform([data['crop']])[0]
        seas = le_season.transform([data['season']])[0]

        features = [[
            int(data['year']),
            loc,
            float(data['area']),
            float(data['rainfall']),
            float(data['temperature']),
            soil,
            irr,
            float(data['humidity']),
            seas,
            crop
        ]]

        prediction = yield_model.predict(features)[0]
        yield_cat  = le_yield_cat.inverse_transform([prediction])[0]

        # Emoji and message based on category
        info = {
            'Low':    {'emoji': '🔴', 'msg': 'Low yield expected. Consider improving soil nutrients or changing crop.'},
            'Medium': {'emoji': '🟡', 'msg': 'Medium yield expected. Good conditions — maintain current practices.'},
            'High':   {'emoji': '🟢', 'msg': 'High yield expected! Excellent conditions for this crop.'}
        }

        return jsonify({
            'success':   True,
            'yield_category': yield_cat,
            'emoji':     info[yield_cat]['emoji'],
            'message':   info[yield_cat]['msg']
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# API 3 — Price Prediction
@app.route('/api/price', methods=['POST'])
def api_price():
    try:
        data = request.json

        loc  = le_location.transform([data['location']])[0]
        soil = le_soil.transform([data['soil']])[0]
        irr  = le_irrigation.transform([data['irrigation']])[0]
        crop = le_crop.transform([data['crop']])[0]
        seas = le_season.transform([data['season']])[0]

        features = [[
            int(data['year']),
            loc,
            float(data['area']),
            float(data['rainfall']),
            float(data['temperature']),
            soil,
            irr,
            float(data['humidity']),
            seas,
            crop
        ]]

        # Predict log price then convert back to actual price
        log_price    = price_model.predict(features)[0]
        actual_price = int(np.expm1(log_price))

        # Give buying/selling advice
        if actual_price > 150000:
            advice = "📈 Great time to sell! Price is expected to be HIGH."
            trend  = "High"
        elif actual_price > 50000:
            advice = "📊 Average market price. Sell if storage costs are high."
            trend  = "Medium"
        else:
            advice = "📉 Price is LOW. Consider storing crop and waiting."
            trend  = "Low"

        return jsonify({
            'success':        True,
            'predicted_price': f"₹{actual_price:,}",
            'trend':          trend,
            'advice':         advice
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── HELPER ROUTE — Get dropdown options ───────
@app.route('/api/options', methods=['GET'])
def api_options():
    return jsonify({
        'locations':   list(le_location.classes_),
        'soils':       list(le_soil.classes_),
        'irrigations': list(le_irrigation.classes_),
        'crops':       list(le_crop.classes_),
        'seasons':     list(le_season.classes_)
    })


# ── RUN APP ───────────────────────────────────
if __name__ == '__main__':
    print("🚀 AgriTech server starting...")
    print("🌐 Open your browser and go to: http://127.0.0.1:5000")
    app.run(debug=True)