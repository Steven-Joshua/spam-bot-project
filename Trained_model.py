import joblib

# Load the model and vectorizer
loaded_model = joblib.load("spam_classifier_model.pkl")
loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Define optimal threshold from training phase
optimal_threshold = 0.24  

# Example text
sample_text = ["Your account has been flagged for suspicious activity. Please verify your identity immediately to avoid suspension. Click here: [malicious link]"]

# Transform text
sample_tfidf = loaded_vectorizer.transform(sample_text)

# Get spam probability
spam_probability = loaded_model.predict_proba(sample_tfidf)[:, 1][0]  # Extract probability for spam class

# Classify based on the custom threshold
prediction = "Spam" if spam_probability > optimal_threshold else "Ham"

# Output results
print(f"Spam Probability: {spam_probability:.2f}")
print("Prediction:", prediction)
