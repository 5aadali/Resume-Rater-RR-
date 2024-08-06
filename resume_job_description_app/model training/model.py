import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load your dataset
df = pd.read_csv('gpt_dataset.csv')  # Adjust the path and filename as necessary

# Print the column names to verify
print("Column names in the dataset:", df.columns)

# Ensure that your dataset contains 'Resume' and 'Category' columns
assert 'Resume' in df.columns, "'Resume' column is missing in the dataset"
assert 'Category' in df.columns, "'Category' column is missing in the dataset"

# Data preprocessing and model training
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X = tfidf.fit_transform(df['Resume'])
y = df['Category']

clf = LogisticRegression()
clf.fit(X, y)

# Save the model and vectorizer
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
pickle.dump(clf, open('clf.pkl', 'wb'))