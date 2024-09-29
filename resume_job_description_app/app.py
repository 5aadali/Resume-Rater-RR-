import os
import re
import csv
from flask import Flask, request, render_template
import pickle
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename

app = Flask(__name__)

base_dir = os.path.dirname(os.path.abspath(__file__))
tfidf_path = os.path.join(base_dir, 'Old models', 'tfidf.pkl')
clf_path = os.path.join(base_dir, 'Old models', 'clf.pkl')

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

def save_to_csv(data):
    csv_file = 'cv_sections.csv'
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as csvfile:
        fieldnames = ['Name', 'Category', 'Education', 'Experience', 'Skills', 'Projects', 'Probability Score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)

def extract_section(cv_text, section_name):
    pattern = re.compile(
        rf'(?i)^{section_name}.*?^(Experience|Skills|Projects|Certifications|Languages|$)',
        re.MULTILINE | re.DOTALL
    )
    match = pattern.search(cv_text)
    if match:
        section = match.group(0).strip()
        next_section = re.split(r'^(Experience|Skills|Projects|Certifications|Languages)', section, maxsplit=1)
        return next_section[0].strip() if next_section else section
    return f"{section_name} section not found."

def extract_section1(cv_text, section_name):
    pattern = re.compile(
        rf'(?i)^{section_name}.*?^(Experience|Projects|Certifications|Languages|$)',
        re.MULTILINE | re.DOTALL
    )
    match = pattern.search(cv_text)
    if match:
        section = match.group(0).strip()
        next_section = re.split(r'^(Experience|Projects|Certifications|Languages)', section, maxsplit=1)
        return next_section[0].strip() if next_section else section
    return f"{section_name} section not found."

def extract_section2(cv_text, section_name):
    pattern = re.compile(
        rf'(?i)^{section_name}.*?^(Certifications|Languages|$)',
        re.MULTILINE | re.DOTALL
    )
    match = pattern.search(cv_text)
    if match:
        section = match.group(0).strip()
        next_section = re.split(r'^(Certifications|Languages)', section, maxsplit=1)
        return next_section[0].strip() if next_section else section
    return f"{section_name} section not found."

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
    resume_sections = {
        'Education': extract_section(resume_text, 'Education'),
        'Experience': extract_section(resume_text, 'Experience'),
        'Skills': extract_section1(resume_text, 'Skills'),
        'Projects': extract_section2(resume_text, 'Projects')
    }

    
    print("Education Section:", resume_sections['Education'])
    print("Experience Section:", resume_sections['Experience'])
    print("Skills Section:", resume_sections['Skills'])
    print("Projects Section:", resume_sections['Projects'])

    job_description_text = resume_sections.get('Education', '') 
    cleaned_job_description = clean_resume(job_description_text) if job_description_text else ''
    
    resume_features = tfidf.transform([cleaned_resume])
    prediction_id = clf.predict(resume_features)[0]
    match_rating, similarity_score = rate_resume_similarity(resume_features, cleaned_job_description)
    rounded_similarity_score = round(similarity_score, 2)

    name = "Unknown"  
    probability_score = rounded_similarity_score
    csv_data = {
        'Name': name,
        'Category': prediction_id,
        'Education': resume_sections['Education'],
        'Experience': resume_sections['Experience'],
        'Skills': resume_sections['Skills'],
        'Projects': resume_sections['Projects'],
        'Probability Score': probability_score
    }
    save_to_csv(csv_data)

    # Prepare CSV data for rendering
    csv_data_for_template = [
        {'Section': 'Education', 'Content': resume_sections['Education']},
        {'Section': 'Experience', 'Content': resume_sections['Experience']},
        {'Section': 'Skills', 'Content': resume_sections['Skills']},
        {'Section': 'Projects', 'Content': resume_sections['Projects']}
    ]

    return render_template('index.html', 
                           prediction_category=prediction_id,
                           resume_match_rating=match_rating,
                           numerical_similarity_score=rounded_similarity_score,
                           csv_data=csv_data_for_template)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
