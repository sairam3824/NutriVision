from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = load_model('model/food_model.h5')

# Load label map
with open('model/labels.json') as f:
    label_map = json.load(f)

# Reverse label mapping
reverse_labels = {v: k for k, v in label_map.items()}

# Load CSV nutrition data
df = pd.read_csv('nutrients_csvfile.csv')

# Clean and normalize CSV values
def safe_float(val):
    try:
        return float(val)
    except:
        return 0.0

def normalize(name):
    return name.strip().lower().replace(" ", "_")

# Build lookup dictionary for nutrients
nutrition_data = {}
for _, row in df.iterrows():
    food_name = normalize(row['Food'])
    nutrition_data[food_name] = {
        'calories': safe_float(row['Calories']),
        'protein': safe_float(row['Protein']),
        'carbs': safe_float(row.get('Carbs', 0)),
        'fat': safe_float(row['Fat']),
    }

# Model prediction function
def model_predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    return reverse_labels[class_idx]

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    if file:
        # Save uploaded image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Predict food item
        predicted_label = model_predict(filepath)
        food_key = normalize(predicted_label)

        # Lookup nutrition info
        macros = nutrition_data.get(food_key, {
            'calories': 'Unknown',
            'protein': 'Unknown',
            'carbs': 'Unknown',
            'fat': 'Unknown'
        })

        # Return result to UI
        return render_template('index.html', result={
            'food': predicted_label.replace('_', ' ').title(),
            'calories': macros['calories'],
            'protein': macros['protein'],
            'carbs': macros['carbs'],
            'fat': macros['fat'],
            'image': filename
        })

    return render_template('index.html', result=None)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
