from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import pickle
import numpy as np
import json
import io
import os

app = Flask(__name__)

# ── LOAD ALL MODELS ───────────────────────────
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

disease_model = load_model('models/disease_model.h5')
with open('models/disease_labels.json') as f:
    disease_labels = json.load(f)

print("✅ All models loaded!")

# ── PAGE ROUTES ───────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend')
def recommend():
    return render_template('recommend.html')

@app.route('/yield')
def yield_page():
    return render_template('yield.html')

@app.route('/price')
def price():
    return render_template('price.html')

@app.route('/disease')
def disease():
    return render_template('disease.html')

# ── API ROUTES ────────────────────────────────

# API 1 — Crop Recommendation
@app.route('/api/recommend', methods=['POST'])
def api_recommend():
    try:
        data = request.json

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

        proba      = crop_model.predict_proba(features)[0]
        top3_idx   = proba.argsort()[-3:][::-1]
        top3_crops = [
            {
                'crop':       le_crop.inverse_transform([i])[0],
                'confidence': round(proba[i] * 100, 1)
            }
            for i in top3_idx
        ]

        return jsonify({
            'success':          True,
            'recommended_crop': crop_name,
            'top3':             top3_crops
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

        info = {
            'Low':    {'emoji': '🔴', 'msg': 'Low yield expected. Consider improving soil nutrients or changing crop.'},
            'Medium': {'emoji': '🟡', 'msg': 'Medium yield expected. Good conditions — maintain current practices.'},
            'High':   {'emoji': '🟢', 'msg': 'High yield expected! Excellent conditions for this crop.'}
        }

        return jsonify({
            'success':        True,
            'yield_category': yield_cat,
            'emoji':          info[yield_cat]['emoji'],
            'message':        info[yield_cat]['msg']
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

        log_price    = price_model.predict(features)[0]
        actual_price = int(np.expm1(log_price))

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
            'success':         True,
            'predicted_price': f"₹{actual_price:,}",
            'trend':           trend,
            'advice':          advice
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# API 4 — Disease Detection
@app.route('/api/disease', methods=['POST'])
def api_disease():
    try:
        file    = request.files['image']
        img     = Image.open(file.stream).convert('RGB').resize((224, 224))
        img_arr = np.array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)

        predictions = disease_model.predict(img_arr)
        idx         = np.argmax(predictions[0])
        confidence  = float(np.max(predictions[0])) * 100
        label       = disease_labels[str(idx)]

        parts   = label.split('___')
        plant   = parts[0].replace('_', ' ')
        disease = parts[1].replace('_', ' ') if len(parts) > 1 else 'Healthy'

        remedies = {
            'healthy':        '✅ Your plant is healthy! Keep maintaining good farming practices.',
            'early blight':   '💊 Apply copper-based fungicide. Remove infected leaves immediately.',
            'late blight':    '💊 Use Mancozeb fungicide. Avoid overhead irrigation.',
            'leaf mold':      '💊 Improve ventilation. Apply fungicide spray weekly.',
            'bacterial spot': '💊 Use copper hydroxide spray. Remove infected parts.',
            'black rot':      '💊 Apply Bordeaux mixture. Prune infected branches.',
            'apple scab':     '💊 Apply fungicide at bud break. Remove fallen leaves.',
            'powdery mildew': '💊 Apply sulfur-based fungicide. Ensure good air circulation.',
            'rust':           '💊 Apply tebuconazole fungicide. Avoid wetting leaves.',
            'leaf spot':      '💊 Apply mancozeb spray. Improve field drainage.',
        }

        remedy = '💊 Consult your local agricultural officer for treatment advice.'
        for key in remedies:
            if key in disease.lower():
                remedy = remedies[key]
                break

        return jsonify({
            'success':    True,
            'plant':      plant,
            'disease':    disease,
            'confidence': round(confidence, 1),
            'remedy':     remedy,
            'label':      label
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ── OPTIONS HELPER ────────────────────────────
@app.route('/api/options', methods=['GET'])
def api_options():
    return jsonify({
        'locations':   list(le_location.classes_),
        'soils':       list(le_soil.classes_),
        'irrigations': list(le_irrigation.classes_),
        'crops':       list(le_crop.classes_),
        'seasons':     list(le_season.classes_)
    })


# ── RUN ───────────────────────────────────────
if __name__ == '__main__':
    print("🚀 AgriTech server starting...")
    print("🌐 Open your browser: http://127.0.0.1:5000")
    app.run(debug=True)
