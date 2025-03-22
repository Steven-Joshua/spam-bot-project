import cv2
import pytesseract
import joblib
import os
import re
import numpy as np
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from db import insert_message  # Import database logging function

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load Model & Vectorizer with Error Handling
try:
    loaded_model = joblib.load("spam_classifier_model.pkl")
    loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")
except FileNotFoundError:
    raise FileNotFoundError("Model or vectorizer file is missing. Ensure they are present before running the app.")

# Optimal spam classification threshold
optimal_threshold = 0.24  

# Configure Tesseract (Ensure this path is correct)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = Flask(__name__)
CORS(app)  # Allow frontend requests

# Limit file upload size (5MB)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

def preprocess_image(image_path):
    """Preprocess image for better OCR accuracy"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Error loading image. Check file path and format.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imwrite("processed_image.png", thresh)  # Debugging
    return thresh

def extract_text_from_image(image_path):
    """Extracts text from an image using OCR"""
    processed_image = preprocess_image(image_path)
    extracted_text = pytesseract.image_to_string(processed_image)
    if not extracted_text.strip():
        return "No readable text found."
    return clean_extracted_text(extracted_text)

def clean_extracted_text(text):
    """Cleans extracted text for better classification"""
    text = text.replace("\n", " ").strip()
    text = ' '.join(text.split())
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
    text = data.get("text", "").strip()
    source = data.get("source", "").strip() or "Unknown"

    if not text:
        return jsonify({"error": "Text input is required."}), 400

    spam_probability, prediction = classify_text(text)
    insert_message(source, text, spam_probability, prediction)

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

    try:
        extracted_text = extract_text_from_image(image_path)
    except Exception as e:
        os.remove(image_path)
        return jsonify({"error": str(e)}), 500

    os.remove(image_path)

    if extracted_text in ["No readable text found.", "OCR Error"]:
        return jsonify({"message": "Image processed, but no readable text detected."})

    spam_probability, prediction = classify_text(extracted_text)
    insert_message(source, extracted_text, spam_probability, prediction)

    return jsonify({
        "Source": source,
        "Extracted Text": extracted_text,
        "Spam Probability": round(spam_probability, 2),
        "Prediction": prediction
    })

@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Error: {e}")
    return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=True)
