import os
from flask import Flask, request, render_template, jsonify
import pickle
import pdfplumber
import re
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define paths for the model and vectorizer files
base_dir = os.path.dirname(os.path.abspath(__file__))
tfidf_path = os.path.join(base_dir, 'Old models', 'tfidf.pkl')
clf_path = os.path.join(base_dir, 'Old models', 'clf.pkl')

# Load the TF-IDF vectorizer and model
tfidf = pickle.load(open(tfidf_path, 'rb'))
clf = pickle.load(open(clf_path, 'rb'))

def pdf_to_text(file_path):
    with pdfplumber.open(file_path) as pdf:
        text = ''
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + '\n'
        return text

def extract_resume_sections(resume_text):
    sections = {}
    job_description_pattern = r'(?<=Job Description:)\s*(.*?)(?=\n|$)'
    skills_pattern = r'(?<=Skills:)\s*(.*?)(?=\n|$)'
    experience_pattern = r'(?<=Experience:)\s*(.*?)(?=\n|$)'
    interest_pattern = r'(?<=Interests:)\s*(.*?)(?=\n|$)'
    
    job_description = re.search(job_description_pattern, resume_text, re.IGNORECASE | re.DOTALL)
    skills = re.search(skills_pattern, resume_text, re.IGNORECASE | re.DOTALL)
    experience = re.search(experience_pattern, resume_text, re.IGNORECASE | re.DOTALL)
    interests = re.search(interest_pattern, resume_text, re.IGNORECASE | re.DOTALL)
    
    sections['job_description'] = job_description.group(1).strip() if job_description else None
    sections['skills'] = skills.group(1).strip() if skills else None
    sections['experience'] = experience.group(1).strip() if experience else None
    sections['interests'] = interests.group(1).strip() if interests else None
    
    return sections

def clean_resume(txt):
    clean_text = re.sub(r'http\S+\s', ' ', txt)
    clean_text = re.sub(r'RT|cc', ' ', clean_text)
    clean_text = re.sub(r'#\S+\s', ' ', clean_text)
    clean_text = re.sub(r'@\S+', ' ', clean_text)
    clean_text = re.sub(r'[%s]' % re.escape('"""#$%&\'()*+,-./:;<=>?@[\]^_\'{|}~'), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

def rate_resume_similarity(resume_features, job_description_text):
    job_description_vector = tfidf.transform([job_description_text])
    similarity = cosine_similarity(resume_features, job_description_vector)
    similarity_score = similarity[0][0]
    if similarity_score >= 0.75:
        return 'High Match', similarity_score
    elif similarity_score >= 0.50:
        return 'Medium Match', similarity_score
    else:
        return 'Low Match', similarity_score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'resume' not in request.files or ('job_description' not in request.files and 'job_description_text' not in request.form and 'job_keywords' not in request.form):
        return render_template('index.html', error='No file part')
    
    resume_file = request.files['resume']
    job_description_file = request.files['job_description']
    job_description_text = request.form.get('job_description_text')
    job_keywords = request.form.get('job_keywords')
    
    if resume_file.filename == '' or (job_description_file.filename == '' and not job_description_text and not job_keywords):
        return render_template('index.html', error='No selected file')
    
    resume_filename = secure_filename(resume_file.filename)
    job_description_filename = secure_filename(job_description_file.filename) if job_description_file else None
    
    resume_path = os.path.join('uploads', resume_filename)
    job_description_path = os.path.join('uploads', job_description_filename) if job_description_file else None
    
    resume_file.save(resume_path)
    if job_description_file:
        job_description_file.save(job_description_path)
    
    resume_text = pdf_to_text(resume_path)
    cleaned_resume = clean_resume(resume_text)
    resume_sections = extract_resume_sections(resume_text)
    job_description_text = resume_sections['job_description']
    skills_text = resume_sections['skills']
    experience_text = resume_sections['experience']
    interests_text = resume_sections['interests']

    cleaned_job_description = clean_resume(job_description_text) if job_description_text else ''
    cleaned_skills = clean_resume(skills_text) if skills_text else ''
    cleaned_experience = clean_resume(experience_text) if experience_text else ''
    cleaned_interests = clean_resume(interests_text) if interests_text else ''

    resume_features = tfidf.transform([cleaned_resume])
    prediction_id = clf.predict(resume_features)[0]
    match_rating, similarity_score = rate_resume_similarity(resume_features, cleaned_job_description)
    rounded_similarity_score = round(similarity_score, 2)

    return render_template('index.html', 
                           prediction_category=prediction_id,
                           resume_match_rating=match_rating,
                           numerical_similarity_score=rounded_similarity_score,
                           job_description=job_description_text,
                           skills=skills_text,
                           experience=experience_text,
                           interests=interests_text)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
