import tkinter as tk
from tkinter import messagebox
import joblib
import pickle
import re

# Load Model and Vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Preprocess Function
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Debugging: Print Model and Vectorizer Details
print("Model Class:", type(model))
print("Vectorizer Class:", type(vectorizer))
print("Vectorizer Vocabulary Size:", len(vectorizer.vocabulary_))

# Create Main Window
root = tk.Tk()
root.title("Fake News Detection System")
root.geometry("600x500")
root.configure(bg="#2c3e50")  # Dark background color

# Heading Label
title_label = tk.Label(root, text="Fake News Detector", font=("Arial", 20, "bold"), bg="#2c3e50", fg="white")
title_label.pack(pady=10)

# Input Text Box
input_text = tk.Text(root, height=8, width=60, font=("Arial", 12), wrap="word", bg="#ecf0f1")
input_text.pack(pady=10)

# Function to Predict Fake or Real News
def check_news():
    news = input_text.get("1.0", "end").strip()
    if not news:
        messagebox.showerror("Error", "Please enter news content!")
        return

    # Preprocess Input Text
    news = preprocess_text(news)
    transformed_text = vectorizer.transform([news])
    prediction = model.predict(transformed_text)[0]
    prediction_proba = model.predict_proba(transformed_text)[0]

    # Debugging Output
    print("Transformed Text Shape:", transformed_text.shape)
    print("Prediction Probabilities:", prediction_proba)
    print("Prediction:", "Fake" if prediction == 1 else "Real")

    if prediction == 0:
        result_label.config(text="🟢 Real News", fg="green")
    else:
        result_label.config(text="🔴 Fake News", fg="red")

# Styled Check Button
check_button = tk.Button(root, text="Check News", font=("Arial", 14, "bold"), bg="#27ae60", fg="white",
                         activebackground="#2ecc71", padx=20, pady=5, command=check_news)
check_button.pack(pady=10)

# Result Label
result_label = tk.Label(root, text="", font=("Arial", 16, "bold"), bg="#2c3e50")
result_label.pack(pady=20)

# Run Tkinter App
root.mainloop()
