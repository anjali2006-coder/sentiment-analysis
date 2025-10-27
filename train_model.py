#pip install pandas numpy sckit-learn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.metrics import accuracy_score, classification_report   
import joblib
import os

#dataset
df = pd.read_csv('dataset/my_dataset.csv')
# print("Total Duplicates:", df.duplicated().sum())
df = df.drop_duplicates()
# print("Total Duplicates:", df.duplicated().sum())

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    text = text.lower().split()  # Lowercase and tokenize
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

df['statements'] = df['statements'].astype(str).apply(preprocess_text)
# print("done")

#text vectorization
X = df['statements']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

lr_model = LogisticRegression(max_iter=1000, solver='liblinear')
lr_model.fit(X_train_vect, y_train)

# Predict
y_pred_lr = lr_model.predict(X_test_vect)

# Evaluate
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Report:\n", classification_report(y_test, y_pred_lr))

# save model 

os.makedirs('model', exist_ok=True)

joblib.dump(vectorizer, 'model/tfidf_vectorizer.pkl')
joblib.dump(lr_model, 'model/logistic_regression_model.pkl')

print("done 2")