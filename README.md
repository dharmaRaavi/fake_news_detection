# 📰 Fake News Detection System

A desktop-based **Fake News Detection System** built using **Python**, **Tkinter**, **Scikit-learn**, and **Natural Language Processing (NLP)**.
This project uses a pre-trained **Machine Learning Model** and **TF-IDF Vectorizer** to classify news articles as **Real** or **Fake**.

---

## 📌 Features
- **User-Friendly GUI** using Tkinter.
- **Real-time News Classification** — Detects if news is *Fake* or *Real*.
- **Pre-trained ML Model Integration**.
- **Text Preprocessing** for better accuracy.
- **Styled Interface** with Dark Theme UI.

---

## 🛠️ Technologies Used
- **Python 3.x**
- **Tkinter** (GUI library)
- **Scikit-learn** (Machine Learning)
- **Joblib & Pickle** (Model serialization)
- **Regular Expressions (re)** (Text preprocessing)

---

## 📂 Project Structure
```
Fake-News-Detection/
│
├── fake_news_model.pkl        # Trained ML model
├── tfidf_vectorizer.pkl       # Trained TF-IDF vectorizer
├── fake_news_gui.py           # Main GUI application
├── README.md                  # Documentation
└── dataset/                   # Dataset folder (e.g., fake.csv)
```

---

## 📊 Dataset Used
You can use the **Fake News Dataset** (Kaggle) or any labeled dataset containing:

| Column       | Description                     |
|--------------|---------------------------------|
| `title`      | Title of the news article       |
| `text`       | Full news content               |
| `label`      | 0 = Real, 1 = Fake               |

Example Dataset Source:  
[Fake News Dataset on Kaggle](https://www.kaggle.com/c/fake-news/data)

---

## ⚙️ How to Run the Project

### **1️⃣ Install Required Libraries**
```bash
pip install tkinter scikit-learn joblib
```

---

### **2️⃣ Train Model (If You Don’t Have .pkl Files)**
If you already have `fake_news_model.pkl` and `tfidf_vectorizer.pkl`, skip this step.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import pickle
import re

# Load Dataset
df = pd.read_csv("fake.csv")

# Combine title and text
df['content'] = df['title'] + " " + df['text']

# Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

df['content'] = df['content'].apply(preprocess_text)

# Features & Labels
X = df['content']
y = df['label']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Save Model & Vectorizer
joblib.dump(model, "fake_news_model.pkl")
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("Model & Vectorizer Saved Successfully!")
```

---

### **3️⃣ Run the GUI Application**
```bash
python fake_news_gui.py
```

---

## 🔍 How It Works
1. User enters news text in the input box.
2. The text is **preprocessed** (lowercased, punctuation removed).
3. The text is **converted to numerical features** using the pre-trained TF-IDF vectorizer.
4. The **ML model** predicts whether the news is *Fake* or *Real*.
5. Result is displayed in the GUI with a colored label.

---

## 📈 Future Improvements
- Add probability/confidence percentage in the result.
- Support multiple languages.
- Add a database for storing checked news history.
- Deploy as a **Web App** using Flask or Django.

---


