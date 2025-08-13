import os
import numpy as np
from flask import Flask, render_template, request, jsonify, flash
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import base64
import io

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained model
try:
    model = load_model('cifar10_advanced_model.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for CIFAR-10 model input"""
    try:
        # Load and resize image to 32x32 (CIFAR-10 input size)
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((32, 32))
        
        # Convert to numpy array and normalize
        img_array = np.array(img)
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_image(image_path):
    """Make prediction on the uploaded image"""
    if model is None:
        return None, "Model not loaded"
    
    try:
        # Preprocess the image
        processed_img = preprocess_image(image_path)
        if processed_img is None:
            return None, "Error preprocessing image"
        
        # Make prediction
        predictions = model.predict(processed_img)
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        top_predictions = []
        
        for idx in top_indices:
            class_name = CIFAR10_CLASSES[idx]
            confidence = float(predictions[0][idx])
            top_predictions.append({
                'class': class_name,
                'confidence': confidence,
                'percentage': round(confidence * 100, 2)
            })
        
        return top_predictions, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Check file size
            file.seek(0, os.SEEK_END)
            file_size = file.tell()
            file.seek(0)
            
            if file_size > MAX_FILE_SIZE:
                return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 400
            
            # Save file
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Make prediction
            predictions, error = predict_image(filepath)
            
            if error:
                return jsonify({'error': error}), 500
            
            # Convert image to base64 for display
            with open(filepath, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'predictions': predictions,
                'image_data': img_data,
                'filename': filename
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, GIF, BMP'}), 400

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
