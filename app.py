import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pdfplumber
import re
import tensorflow as tf
import pickle

MAX_SEQ_LEN=200

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

def load_tokenizer_and_labels():
    with open("tokenizer (1).pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("labels.pkl", "rb") as f:
        le = pickle.load(f)
    return tokenizer, le

model = load_model()
tokenizer, le = load_tokenizer_and_labels()

st.title('ML Based Resume Screener')

roles=['Data Science', 'HR', 'Advocate', 'Arts', 'Web Designing',
       'Mechanical Engineer', 'Sales', 'Health and fitness',
       'Civil Engineer', 'Java Developer', 'Business Analyst',
       'SAP Developer', 'Automation Testing', 'Electrical Engineering',
       'Operations Manager', 'Python Developer', 'DevOps Engineer',
       'Network Security Engineer', 'PMO', 'Database', 'Hadoop',
       'ETL Developer', 'DotNet Developer', 'Blockchain', 'Testing']
target=st.selectbox("Select the Role You're Hiring For:", roles)

uploaded_file=st.file_uploader("Upload Resume(pdf)", type=["pdf"])

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        text=''.join(page.extract_text() for page in pdf.pages)
    
    st.subheader('Resume Preview')
    st.write(text[:1000])
    cleaned=clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    predicted=model.predict(padded)
    predicted_index = predicted.argmax(axis=1)[0]
    predicted_label = le.inverse_transform([predicted_index])[0]


    st.write(f'Predicted Resume Category: {predicted_label}')
    if predicted_label==target:
        st.success("Candidate is suitable for the role")
    else:
        st.warning(f'Candidate appears more suitable for {predicted_label} role')

#extracting sections
def extract_sections(text,section_list):
    sections={}
    pattern='|'.join([re.escape(s) for s in section_list])
    matches=list(re.finditer(rf"(?i)\b({pattern})\b",text))
    for i in range(len(matches)):
        start=matches[i].start()
        end=matches[i+1].start() if i+1<len(matches) else len(text)
        section = matches[i].group().strip().capitalize()
        sections[section] = text[start:end].strip()

    return sections
if uploaded_file:
    raw_text = text
    
    sections = extract_sections(raw_text, [
        
    "Education", "Work Experience", "Professional Experience",
    "Skills", "Projects", "Certifications", "Achievements",
    "Objective", "Summary", "Internships", "Technologies", "Technical skills"
    ])

    for section, content in sections.items():
        st.subheader(section)
        st.text_area("", content[:1000], height=150)

        