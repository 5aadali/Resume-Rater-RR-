import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Check the current working directory
print("Current Working Directory:", os.getcwd())

# List all files in the current directory
files = os.listdir()
print("Files in the current directory:", files)

# Define the file path
file_path = 'resume_ds.csv'  # Adjust the path and filename as necessary

# Ensure that the file exists
if not os.path.exists(file_path):
    print(f"File '{file_path}' not found.")
else:
    print(f"File '{file_path}' found.")

    # Load the dataset
    df = pd.read_csv(file_path)
    print("Loaded dataset:")
    print(df.head())

    # Print the column names to verify
    print("Column names in the dataset:", df.columns)

    # Ensure that your dataset contains 'Resume' and 'Category' columns
    if 'Resume' not in df.columns:
        print("'Resume' column is missing in the dataset")
    if 'Category' not in df.columns:
        print("'Category' column is missing in the dataset")

    # Data preprocessing and model training
    print("Starting TF-IDF vectorization...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X = tfidf.fit_transform(df['Resume'])
    y = df['Category']

    # Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training data shape:", X_train.shape)
    print("Testing data shape:", X_test.shape)

    # Train the model
    print("Training the model...")
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # Predict and evaluate
    print("Making predictions...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Save the model and vectorizer
    print("Saving model and vectorizer...")
    pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
    pickle.dump(clf, open('clf.pkl', 'wb'))
    print("Model and vectorizer saved successfully.")
