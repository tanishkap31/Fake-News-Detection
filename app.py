from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import string
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

#import nltk
#nltk.download('stopwords')


# ----------------------------------
# Flask App Setup
# ----------------------------------
app = Flask(__name__)

# ----------------------------------
# Data & Model Setup (Run Once)
# ----------------------------------
true_df = pd.read_csv("TrueNews.csv")
fake_df = pd.read_csv("FakeNews.csv")

true_df["label"] = 1
fake_df["label"] = 0

df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)

# Preprocessing
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df["content"] = df["title"] + " " + df["text"]
df["content"] = df["content"].apply(clean_text)

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# ----------------------------------
# Routes
# ----------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news_text = request.form["news"]
    cleaned = clean_text(news_text)
    tfidf = vectorizer.transform([cleaned])
    prediction = model.predict(tfidf)[0]
    probability = model.predict_proba(tfidf)[0][prediction]

    result_label = "Real" if prediction == 1 else "Fake"
    css_class = "Real" if prediction == 1 else "Fake"

    # Graphs
    y_pred = model.predict(X_test_tfidf)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fake", "Real"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("static/confusion.png")
    plt.close()

    # F1 Score Bar Plot
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose().drop(["accuracy", "macro avg", "weighted avg"])
    plt.figure(figsize=(6, 4))
    sns.barplot(x=report_df.index, y=report_df["f1-score"])
    plt.title("F1 Score by Class")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1)
    plt.savefig("static/f1score.png")
    plt.close()

    # Prediction Probability Pie
    proba = model.predict_proba(tfidf)[0]
    plt.pie(proba, labels=["Fake", "Real"], autopct='%1.1f%%', startangle=90, colors=['red', 'green'])
    plt.title("Prediction Probability")
    plt.savefig("static/probability.png")
    plt.close()

    return render_template("result.html", 
                           input_text=news_text,
                           prediction=result_label,
                           prediction_class=css_class,
                           probability=round(probability * 100, 2),
                           show_graphs=True)

# ----------------------------------
# Run the App
# ----------------------------------
if __name__ == "__main__":
    app.run(debug=True)
