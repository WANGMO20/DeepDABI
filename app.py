import os
# Set TensorFlow logging level to suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the model and class names
model = tf.keras.models.load_model('model.h5', compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define the class labels as a list where index matches the prediction index
class_names = [
    "Ayrshire Cattle",
    "Brown Swiss Cattle",
    "Holstein Friesian Cattle",
    "Jersey Cattle",
    "Red Dane Cattle",
    "Bengal",
    "Domestic Shorthair",
    "French Bulldog",
    "German Shepherd",
    "Golden Retriever",
    "Maine Coon",
    "Poodle",
    "Ragdoll",
    "Siamese",
    "Yorkshire Terrier"
]

def is_valid_image(file_stream):
    try:
        # Try to open the image using PIL
        image = Image.open(file_stream)
        # Verify it's actually an image by attempting to load it
        image.verify()
        file_stream.seek(0)  # Reset file pointer
        return True
    except Exception:
        return False

def process_image(file_stream):
    try:
        # Open image using PIL
        image = Image.open(file_stream)
        
        # Convert to RGB if necessary (handles PNG, RGBA, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        image = image.resize((224, 224))
        
        return image
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def cleanup_old_files():
    """Simple cleanup function to remove old files"""
    try:
        current_time = time.time()
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # Remove files older than 1 hour
            if os.path.getmtime(filepath) < current_time - 3600:
                try:
                    os.remove(filepath)
                except OSError:
                    pass
    except Exception as e:
        print(f"Cleanup error: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    # Run cleanup on each upload request
    cleanup_old_files()
    return render_template('upload.html')

@app.route('/uploaded', methods=['POST'])
def uploaded():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No file selected', 400
    
    # Check if it's a valid image
    if not is_valid_image(file):
        return 'Invalid image file', 400
    
    try:
        # Process the image
        file.seek(0)  # Reset file pointer
        image = process_image(file)
        
        # Generate a secure filename with .jpg extension
        filename = secure_filename(os.path.splitext(file.filename)[0] + '.jpg')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Save the processed image
        image.save(filepath, 'JPEG', quality=95)
        
        return render_template('uploaded.html', image_path=f'uploads/{filename}')
    
    except Exception as e:
        return f'Error processing image: {str(e)}', 400

@app.route('/detect', methods=['POST'])
def detect_breed():
    image_path = request.form.get('image_path')
    if not image_path:
        return jsonify({'error': 'No image path provided'}), 400

    try:
        # Load and preprocess the image
        img_path = os.path.join('static', image_path)
        
        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224))
            
            # Convert to numpy array and preprocess
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

        # Get prediction
        predictions = model.predict(img_array)
        
        # Get the predicted class index and confidence score
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        
        # Map index to breed name
        breed = class_names[predicted_class_index]
        
        # Format the breed name and confidence
        breed_result = f"{breed} ({confidence:.2%} confidence)"
        
        return render_template('result.html', breed=breed_result, image_path=image_path)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.after_request
def add_header(response):
    """
    Add headers to prevent caching
    """
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

if __name__ == '__main__':
    app.run(debug=True)
