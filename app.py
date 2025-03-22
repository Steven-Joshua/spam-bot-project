import cv2
import pytesseract
import joblib
import os
import re
import numpy as np
from db import insert_message  # Import database logging function

# Load the Spam Detection Model & Vectorizer
loaded_model = joblib.load("spam_classifier_model.pkl")
loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Optimal spam classification threshold
optimal_threshold = 0.24  

# Set Tesseract path (Update this based on your system)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_image(image_path):
    """Preprocess image for better OCR accuracy"""
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("Error loading image. Check file path and format.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Apply thresholding (from your script)
    thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Save for debugging
    cv2.imwrite("processed_image.png", thresh)

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
    text = text.replace("\n", " ").strip()  # Remove new lines
    text = ' '.join(text.split())  # Remove extra spaces
    text = re.sub(r"[^a-zA-Z0-9\s:/]", "", text)  # Remove special characters except URLs
    text = re.sub(r"\b[a-zA-Z]{1,2}\b", "", text)  # Remove single/double-letter gibberish words
    return text.lower()  # Convert to lowercase

def classify_text(text):
    """Classifies input text as spam or ham"""
    if not text.strip():
        return 0.0, "Invalid or empty text input."

    text_tfidf = loaded_vectorizer.transform([text])
    spam_probability = loaded_model.predict_proba(text_tfidf)[:, 1][0]
    prediction = "Spam" if spam_probability > optimal_threshold else "Ham"
    return spam_probability, prediction

# 🛠️ **API Integration for Text and Image**
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/test-text", methods=["POST"])
def test_text():
    """API endpoint for text spam detection"""
    data = request.json
    text = data.get("text", "").strip()
    source = data.get("source", "").strip() or "Unknown"

    if not text:
        return jsonify({"error": "Text input is required."}), 400

    spam_probability, prediction = classify_text(text)

    # Store result in MySQL
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

    extracted_text = extract_text_from_image(image_path)
    os.remove(image_path)  # Clean up after processing

    if extracted_text in ["No readable text found.", "OCR Error"]:
        return jsonify({"message": "Image processed, but no readable text detected."})

    spam_probability, prediction = classify_text(extracted_text)

    # Store result in MySQL
    insert_message(source, extracted_text, spam_probability, prediction)

    return jsonify({
        "Source": source,
        "Extracted Text": extracted_text,
        "Spam Probability": round(spam_probability, 2),
        "Prediction": prediction
    })

if __name__ == "__main__":
    app.run(debug=True)
