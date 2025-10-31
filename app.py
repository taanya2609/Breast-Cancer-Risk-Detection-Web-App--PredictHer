from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import joblib
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load trained model
model = joblib.load("model.pkl")

# Ensure upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/treatment')
def treatment():
    return render_template('treatment.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/detection', methods=['GET', 'POST'])
def detection():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))
            img = img.flatten().reshape(1, -1)

            # Predict
            prediction = model.predict(img)[0]
            result = "Malignant" if prediction == 1 else "Benign"

            return render_template('detection.html', result=result, image_url=filepath)
    
    return render_template('detection.html')

if __name__ == '__main__':
    app.run(debug=True)
