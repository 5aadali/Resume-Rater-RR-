import pickle
import pdfplumber
import re
from sklearn.metrics.pairwise import cosine_similarity
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Define paths for the model and vectorizer files
tfidf_path = 'tfidf.pkl'
clf_path = 'clf.pkl'

# Load the TF-IDF vectorizer and model
tfidf = pickle.load(open(tfidf_path, 'rb'))
clf = pickle.load(open(clf_path, 'rb'))

def pdf_to_text(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ''
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # Ensure text is not None
                text += page_text
    return text

def clean_resume(txt):
    clean_text = re.sub(r'http\S+\s', ' ', txt)
    clean_text = re.sub(r'RT|cc', ' ', clean_text)
    clean_text = re.sub(r'#\S+\s', ' ', clean_text)
    clean_text = re.sub(r'@\S+', ' ', clean_text)
    clean_text = re.sub(r'[%s]' % re.escape('"""#$%&\'()*+,-./:;<=>?@[\]^_\'{|}~'), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()  # Strip extra whitespace
    return clean_text

def rate_resume_similarity(resume_features, job_description_text):
    # Vectorize the job description
    job_description_vector = tfidf.transform([job_description_text])
    
    # Calculate similarity score
    similarity = cosine_similarity(resume_features, job_description_vector)
    
    # Extract similarity score from the array
    similarity_score = similarity[0][0]
    
    # Define thresholds for rating
    if similarity_score >= 0.75:
        return 'High Match', similarity_score
    elif similarity_score >= 0.50:
        return 'Medium Match', similarity_score
    else:
        return 'Low Match', similarity_score

# Function to select a file using a file dialog
def select_file(title):
    Tk().withdraw()  # Hide the root window
    filename = askopenfilename(title=title, filetypes=[("PDF files", "*.pdf")])
    return filename

# Ask the user to select the resume PDF file
resume_path = select_file("Select the resume PDF file")

# Ask the user to select the job description PDF file
job_description_path = select_file("Select the job description PDF file")

# Extract and clean resume text
resume_text = pdf_to_text(resume_path)
cleaned_resume = clean_resume(resume_text)

# Extract and clean job description text
job_description_text = pdf_to_text(job_description_path)
cleaned_job_description = clean_resume(job_description_text)

# Transform the cleaned resume text
resume_features = tfidf.transform([cleaned_resume])

# Predict the category
prediction_id = clf.predict(resume_features)[0]

# Rate the resume based on job description similarity
match_rating, similarity_score = rate_resume_similarity(resume_features, cleaned_job_description)

# Print the prediction, match rating, and numerical similarity score
print("Prediction category:", prediction_id)
print("Resume match rating:", match_rating)
print("Numerical similarity score:", similarity_score)
