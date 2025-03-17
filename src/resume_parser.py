import pdfplumber
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF resume"""
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

def extract_skills_experience(resume_text):
    """Extract key skills & experience from the resume"""
    doc = nlp(resume_text)
    skills = [ent.text for ent in doc.ents if ent.label_ == "SKILL"]
    experience = [ent.text for ent in doc.ents if ent.label_ in ["DATE", "TIME"]]
    
    return {"skills": skills, "experience": experience}
