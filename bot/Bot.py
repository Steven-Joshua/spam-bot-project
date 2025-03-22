import joblib
import pytesseract
import cv2
import os
import re
import numpy as np
from PIL import Image

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
    
    # Resize for better OCR accuracy
    scale_factor = 1.5
    new_size = (int(gray.shape[1] * scale_factor), int(gray.shape[0] * scale_factor))
    gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_LINEAR)

    # Apply adaptive thresholding (better for uneven lighting)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Remove small noise using morphological opening
    kernel = np.ones((1,1), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    return cleaned

def extract_text_from_image(image_path):
    """Extracts text from an image using OCR"""
    processed_image = preprocess_image(image_path)
    extracted_text = pytesseract.image_to_string(processed_image)

    if not extracted_text.strip():
        return "No readable text found in the image."

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

def spam_detection_bot(input_data):
    """Determines whether input is text or an image and classifies it"""
    
    if isinstance(input_data, str):  
        if os.path.exists(input_data) and input_data.lower().endswith(('.png', '.jpg', '.jpeg')):  
            # Input is an image
            extracted_text = extract_text_from_image(input_data)
            print(f"\nExtracted Text: {extracted_text}")

            if extracted_text == "No readable text found in the image.":
                return extracted_text
            
            spam_probability, prediction = classify_text(extracted_text)
        else:
            # Input is plain text
            spam_probability, prediction = classify_text(input_data)
    
    else:
        return "Invalid input. Please provide either text or an image file."

    return f"Spam Probability: {spam_probability:.2f}\nPrediction: {prediction}"

# -----------------------------------------
# User Input Handling (Interactive CLI)
# -----------------------------------------
while True:
    print("\n--- Spam Detection Bot ---")
    print("1. Enter Text")
    print("2. Upload Image (Type Image Path)")
    print("3. Exit")

    choice = input("Choose an option (1/2/3): ")

    if choice == "1":
        user_text = input("\nEnter the text message: ")
        print("\n[Text Input Classification]")
        print(spam_detection_bot(user_text))

    elif choice == "2":
        user_image = input("\nEnter image file path: ")
        print("\n[Image Input Classification]")
        print(spam_detection_bot(user_image))

    elif choice == "3":
        print("\nExiting... Goodbye! 👋")
        break

    else:
        print("\nInvalid choice! Please select 1, 2, or 3.")
