import pandas as pd
import re
import string
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, 
    precision_recall_curve, roc_curve, auc
)

# Download required NLTK data
nltk.download("stopwords")
nltk.download("wordnet")

# Load dataset
df = pd.read_csv("C:/Users/Joshua/Downloads/Ai/dataset/mail_data_ml.csv", encoding="latin-1")

# Rename columns for clarity
df.columns = ["Category", "Message"]

# Drop missing values
df.dropna(inplace=True)

# Convert labels: ham = 0, spam = 1
df["Category"] = df["Category"].map({"ham": 0, "spam": 1})

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    words = text.split()  # Tokenization
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatization & stopwords removal
    return " ".join(words)

# Apply preprocessing
df["Cleaned_Message"] = df["Message"].apply(preprocess_text)

# Split dataset (Stratified to maintain spam/ham ratio)
X_train, X_test, y_train, y_test = train_test_split(
    df["Cleaned_Message"], df["Category"], test_size=0.2, random_state=42, stratify=df["Category"]
)

# Convert text into numerical representation using TF-IDF
vectorizer = TfidfVectorizer(analyzer="word", max_features=10000, stop_words="english", ngram_range=(1, 3))  # Improved
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 🔹 Check Out-of-Vocabulary (OOV) Words
vectorizer_vocab = set(vectorizer.get_feature_names_out())
oov_words = [word for word in ["spam", "scam", "free", "offer", "discount", "winner"] if word not in vectorizer_vocab]
print("Out-of-Vocabulary Words:", oov_words)

# 🔹 Hyperparameter tuning for best `alpha`
param_grid = {"alpha": [0.05, 0.1, 0.3, 0.5, 1.0, 2.0]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring="recall", n_jobs=-1)
grid_search.fit(X_train_tfidf, y_train)

# Get best model
best_alpha = grid_search.best_params_["alpha"]
model = MultinomialNB(alpha=best_alpha)
model.fit(X_train_tfidf, y_train)

# Predictions & Probabilities
y_pred = model.predict(X_test_tfidf)
y_probs = model.predict_proba(X_test_tfidf)[:, 1]  # Get probabilities for spam class

# 🔹 Adjust decision threshold dynamically using F1-score
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
optimal_threshold = thresholds[np.argmax(f1_scores)]
y_pred_adjusted = (y_probs > optimal_threshold).astype(int)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred_adjusted)
classification_rep = classification_report(y_test, y_pred_adjusted)
conf_matrix = confusion_matrix(y_test, y_pred_adjusted)

print(f"Model Accuracy: {accuracy:.2f}")
print(f"Optimal Threshold: {optimal_threshold:.2f}")
print("\nClassification Report:\n", classification_rep)

# 🔹 Confusion Matrix Visualization
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 🔹 Precision-Recall Curve
plt.figure(figsize=(6, 4))
plt.plot(recalls, precisions, marker="o", color="red")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.show()

# 🔹 ROC Curve & AUC Score
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color="blue", label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# 🔹 Get Top Spam Words (Model Insights)
feature_names = vectorizer.get_feature_names_out()
class_log_prob = model.feature_log_prob_[1]  # Spam class

# 🔹 Filter out single-character words and irrelevant tokens
top_indices = np.argsort(class_log_prob)[-20:]
top_words = [feature_names[i] for i in top_indices if len(feature_names[i]) > 2]  # Ignore single chars

# 🔹 Improved Bar Plot for Top Spam Words
plt.figure(figsize=(8, 6))
sns.barplot(y=top_words, x=class_log_prob[top_indices][-len(top_words):], hue=["Spam"]*len(top_words), palette="Reds_r", legend=False)
plt.xlabel("Log Probability (Spam)")
plt.ylabel("Top Spam Words")
plt.title("Most Influential Words in Spam Detection")
plt.show()

print("Top spam words:", top_words)

# 🔹 Save Model & Vectorizer
joblib.dump(model, "spam_classifier_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")

# 🔹 Test Model on New Samples
def predict_message(msg):
    processed_msg = preprocess_text(msg)  # Preprocess message
    msg_tfidf = vectorizer.transform([processed_msg])  # Convert to TF-IDF
    prob = model.predict_proba(msg_tfidf)[:, 1][0]  # Get spam probability
    prediction = "Spam" if prob > optimal_threshold else "Ham"
    return prediction, prob

# Example Test
test_messages = [
    "Suspected Spam : Gold Appreciation & High Dividends - Dual Benefits! | SEBI & RBI Certified for Safe Investing. Click to Join for Free. https://goo.su/iBUkLO 12:09 pm",
    "Hey, let's meet for coffee tomorrow.",
    "Exclusive offer! Get 50% discount on your next purchase."
]

for msg in test_messages:
    label, prob = predict_message(msg)
    print(f"Message: {msg}\nPrediction: {label} (Spam Probability: {prob:.2f})\n")
