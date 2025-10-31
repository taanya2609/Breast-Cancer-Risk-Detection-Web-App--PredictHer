import os
import cv2
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Define dataset path
DATASET_PATH = "dataset/"
CATEGORIES = ["benign", "malignant"]
IMG_SIZE = 128  # Resize images to 128x128

# Load images and labels
data = []
labels = []

for category in CATEGORIES:
    folder_path = os.path.join(DATASET_PATH, category)
    label = CATEGORIES.index(category)  # Assign 0 for benign, 1 for malignant

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Resize
        data.append(img.flatten())  # Flatten image
        labels.append(label)

# Convert to NumPy arrays
X = np.array(data)
y = np.array(labels)

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
