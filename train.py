import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

X = []
y = []

def process_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Could not read:", path)
        return None

    img = cv2.resize(img, (100, 50))
    img = img.flatten()
    return img

# Real signatures
for file in os.listdir("signatures"):
    path = os.path.join("signatures", file)
    data = process_image(path)
    if data is not None:
        X.append(data)
        y.append(1)

# Fake signatures
for file in os.listdir("test"):
    path = os.path.join("test", file)
    data = process_image(path)
    if data is not None:
        X.append(data)
        y.append(0)

print("Total images loaded:", len(X))

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model trained successfully ✅")