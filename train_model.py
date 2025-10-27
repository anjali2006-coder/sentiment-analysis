#pip install pandas numpy sckit-learn
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import joblib

#dataset
df = pd.read_csv('dataset/my_dataset.csv')