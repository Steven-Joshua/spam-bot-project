from flask import Flask, request, jsonify
import joblib
import pytesseract
import cv2
import os
import re
import numpy as np
from db import insert_message  # Import the database logging function

app = Flask(__name__)

# Load the Spam Detection Model & Vectorizer
loaded_model = joblib.load("spam_classifier_model.pkl")
loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Set optimal spam classification threshold
optimal_threshold = 0.24  

# Set Tesseract path (update based on your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def preprocess_image(image_path):
    """Preprocess image for better OCR accuracy"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error loading image. Check file path.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    kernel = np.ones((1,1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    return cleaned

def extract_text_from_image(image_path):
    """Extract text from an image using OCR"""
    processed_image = preprocess_image(image_path)
    extracted_text = pytesseract.image_to_string(processed_image)
    return clean_extracted_text(extracted_text) if extracted_text.strip() else "No readable text found."

def clean_extracted_text(text):
    """Cleans extracted text for better classification"""
    text = text.replace("\n", " ").strip()
    text = re.sub(r"[^a-zA-Z0-9\s:/]", "", text)
    text = re.sub(r"\b[a-zA-Z]{1,2}\b", "", text)
    return text.lower()

def classify_text(text):
    """Classifies input text as spam or ham"""
    if not text.strip():
        return 0.0, "Invalid or empty text input."
    
    text_tfidf = loaded_vectorizer.transform([text])
    spam_probability = loaded_model.predict_proba(text_tfidf)[:, 1][0]
    prediction = "Spam" if spam_probability > optimal_threshold else "Ham"
    
    return spam_probability, prediction

@app.route("/test-text", methods=["POST"])
def test_text():
    """API endpoint for text spam detection"""
    data = request.json
    text = data.get("text", "")
    source = data.get("source", "Unknown")
    
    spam_probability, prediction = classify_text(text)
    insert_message(source, text, spam_probability, prediction)  # Store in MySQL
    
    return jsonify({
        "Source": source,
        "Message Text": text,
        "Spam Probability": round(spam_probability, 2),
        "Prediction": prediction
    })

@app.route("/test-image", methods=["POST"])
def test_image():
    """API endpoint for image spam detection"""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    source = request.form.get("source", "Unknown")

    image_path = "temp_image.jpg"
    file.save(image_path)

    extracted_text = extract_text_from_image(image_path)
    spam_probability, prediction = classify_text(extracted_text)
    
    os.remove(image_path)  # Clean up
    insert_message(source, extracted_text, spam_probability, prediction)  # Store in MySQL

    return jsonify({
        "Source": source,
        "Extracted Text": extracted_text,
        "Spam Probability": round(spam_probability, 2),
        "Prediction": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)