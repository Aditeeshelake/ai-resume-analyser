import streamlit as st
import PyPDF2
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to clean text
def clean_text(text):
    tokens = nltk.word_tokenize(text.lower())
    return " ".join(tokens)

# Function to calculate similarity
def calculate_similarity(resume, job_desc):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume, job_desc])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return round(similarity[0][0] * 100, 2)

# Basic skill list
skills_db = ["python", "java", "c", "c++", "sql", "html", "css", "javascript", "machine learning", "data analysis"]

def extract_skills(text):
    found_skills = []
    for skill in skills_db:
        if skill in text:
            found_skills.append(skill)
    return found_skills

# Streamlit UI
st.title("📄 AI Resume Analyzer")

uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")
job_desc = st.text_area("Paste Job Description")

if uploaded_file is not None and job_desc:
    resume_text = extract_text_from_pdf(uploaded_file)
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_desc)

    score = calculate_similarity(resume_clean, job_clean)
    skills = extract_skills(resume_clean)

    st.subheader("📊 Match Score")
    st.write(f"{score}%")

    st.subheader("🧠 Extracted Skills")
    st.write(skills)

    st.subheader("💡 Suggestions")
    missing_skills = [skill for skill in skills_db if skill not in skills]
    
    if score < 50:
        st.write("⚠️ Low match! Improve your resume.")
    elif score < 75:
        st.write("👍 Moderate match. Add more relevant skills.")
    else:
        st.write("🔥 Great match!")

    st.write("Consider adding these skills:", missing_skills[:5])
