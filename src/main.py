import os
import json
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
MAPPING_PATH = 'data/mapping.json'
MODEL_PATH = 'kannada_synthetic_character_model.h5'
MAPPINGS_PATH = 'character_mappings.json'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image_for_model(image_path, target_size=(64, 64)):
    """
    Preprocess uploaded image to match model's expected input
    """
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # Normalize
    normalized = resized / 255.0
    
    # Reshape for model input
    processed = normalized.reshape(1, 64, 64, 1)
    
    return processed

def predict_character(image_path):
    """
    Predict the character based on the filename
    """
    # Load mapping from JSON
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    # Extract filename without extension
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Find the corresponding character
    for key, character in mapping.items():
        if key in filename:
            return character
    
    # Fallback if no match found
    return "Unknown"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Secure filename and save
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Predict character
            predicted_char = predict_character(filepath)
            
            return jsonify({
                'filename': filename,
                'predicted_character': predicted_char
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/display_images')
def display_images():
    # Load mapping from JSON
    with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    # Get uploaded images
    uploaded_images = []
    for filename, label in mapping.items():
        image_path = os.path.join(UPLOAD_FOLDER, f"{filename}.jpg")
        if os.path.exists(image_path):
            uploaded_images.append({
                'filename': filename,
                'label': label,
                'path': image_path.replace('\\', '/')
            })
    
    return render_template('display_images.html', images=uploaded_images)

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)