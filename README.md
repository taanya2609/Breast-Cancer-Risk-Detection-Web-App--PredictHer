## **PredictHer – Breast Cancer Risk Analysis**

This project uses mammogram image datasets to classify breast tumors into **benign** or **malignant** categories using a **Random Forest** machine learning model and **OpenCV** for image preprocessing. The web application is built with **Flask** and includes multiple interactive pages, providing users with instant results, treatment guidance, and awareness resources. The system demonstrates the integration of AI, computer vision, and web technologies in medical image analysis for early cancer detection.

---

### **Project Overview**

**Title:**
PredictHer – Breast Cancer Risk Analysis

**Objective:**
To design an intelligent web-based system that analyzes mammogram images using Random Forest and OpenCV to classify tumors as benign or malignant, providing early breast cancer risk assessment and awareness through an interactive interface.

---
### **How to Run the Project**

**Install Requirements**
Ensure Python 3.8+ and pip are installed.
Then install dependencies:

pip install flask opencv-python scikit-learn numpy pandas joblib


**Project Structure**
Keep files organized as:

PredictHer/
├── app.py
├── train_model.py
├── static/style.css
├── templates/
│   ├── index.html
│   ├── detection.html
│   ├── treatment.html
│   ├── about.html
│   └── contact.html
├── dataset/
└── model/


**Train the Model**
Run the training script to generate and save the model:

python train_model.py


**Run the Web App**
Start the Flask server:

python app.py

Open your browser and visit http://127.0.0.1:5000/

Use the App
Upload a mammogram image on the Detection page to get a prediction (Benign or Malignant) and view treatment guidance.

---
### **Key Components**

1. **Image Dataset and Preprocessing**

   * Mammogram images are collected and labeled into benign and malignant categories.
   * OpenCV is used for preprocessing tasks such as grayscale conversion, resizing, noise reduction, and contrast enhancement to improve image quality for feature extraction.

2. **Feature Extraction**

   * Important image features like texture, edge sharpness, and pixel intensity are extracted using OpenCV methods.
   * These features are then converted into numerical values suitable for training the machine learning model.

3. **Model Training (`train_model.py`)**

   * A **Random Forest Classifier** is trained on the extracted image features.
   * The dataset is split into training and testing subsets to validate model performance.
   * The trained model is saved using **joblib** for deployment in the web application.

4. **Web Application Backend (`app.py`)**

   * Built using **Flask**, the backend handles file uploads, routes between pages, and connects with the trained Random Forest model for predictions.
   * Upon uploading an image, the system processes it, predicts the result, and displays the classification (benign or malignant).

5. **Frontend Interface (HTML & CSS Files)**

   * `index.html`: Home page introducing the project.
   * `detection.html`: Image upload and prediction results.
   * `treatment.html`: Provides medical awareness and treatment guidance.
   * `about.html` & `contact.html`: Provide background information and communication options.
   * `style.css`: Ensures a visually appealing and responsive layout for a smooth user experience.

6. **Results and Visualization**

   * Displays classification results along with confidence levels.
   * Provides educational content and links for medical support and awareness.

---

### **Results Summary**

* Successfully classified mammogram images into benign and malignant with high accuracy.
* Demonstrated robustness and interpretability using Random Forest.
* Achieved efficient image preprocessing and feature extraction through OpenCV.
* Offered a simple, fast, and user-friendly diagnosis experience through the web interface.

---

### **Applications**

* Early detection and diagnosis of breast cancer.
* Educational and research purposes in healthcare AI.
* Clinical support systems for radiologists and oncologists.
* Awareness and self-assessment platform for the public.

---

### **Future Enhancements**

* Integration of **deep learning models** (CNN or hybrid Random Forest-CNN).
* Deployment on **cloud platforms** for real-time large-scale accessibility.
* Addition of **patient data analytics** for personalized risk prediction.
* Enhanced visualization dashboards for medical professionals.

---
