import os
import faiss
import chromadb
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from loguru import logger
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Hugging Face API Token (Use from .env file)
HF_TOKEN = os.getenv("HF_TOKEN")

# Define Directories
ROOT_DIR = "/app"
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Define File Paths
JOB_LISTINGS_FILE = os.path.join(DATA_DIR, "job_listings.csv")
HR_POLICIES_FILE = os.path.join(DATA_DIR, "hr_policies.txt")
INTERVIEW_FAQS_FILE = os.path.join(DATA_DIR, "interview_faqs.txt")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")
JOB_TITLES_FILE = os.path.join(DATA_DIR, "job_titles.npy")

# Load Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS Index if exists
if os.path.exists(FAISS_INDEX_FILE):
    faiss_index = faiss.read_index(FAISS_INDEX_FILE)
    job_titles = np.load(JOB_TITLES_FILE, allow_pickle=True).tolist()
else:
    faiss_index = faiss.IndexFlatL2(384)
    job_titles = []

# Load ChromaDB
chroma_client = chromadb.PersistentClient(path=os.path.join(DATA_DIR, "chroma_db"))

# Load LLM (Mistral-7B)
model_name = "mistralai/Mistral-7B-v0.1"  # Public version
llm_model = pipeline("text-generation", model=model_name, token=HF_TOKEN)

# Logging Setup
logger.add("logs/api.log", rotation="10MB", retention="7 days", level="INFO")

# Initialize FastAPI
app = FastAPI()

# -------------------------------
# API Endpoints
# -------------------------------

# 1️⃣ **Add Job Listing**
class JobListing(BaseModel):
    title: str
    description: str

@app.post("/add_job/")
def add_job(job: JobListing):
    """Adds a new job listing to CSV and FAISS index"""
    global job_titles, faiss_index
    embedding = embedding_model.encode(job.description).astype(np.float32)

    # Store in FAISS
    faiss_index.add(np.array([embedding]))
    job_titles.append(job.title)
    faiss.write_index(faiss_index, FAISS_INDEX_FILE)
    np.save(JOB_TITLES_FILE, np.array(job_titles))

    # Store in CSV
    df = pd.DataFrame([[job.title, job.description]], columns=["title", "description"])
    df.to_csv(JOB_LISTINGS_FILE, mode='a', header=not os.path.exists(JOB_LISTINGS_FILE), index=False)

    return {"message": "Job added successfully"}

# 2️⃣ **Upload Resume & Match Jobs**
@app.post("/match_jobs/")
async def match_jobs(file: UploadFile = File(...)):
    """Matches uploaded resume with job listings using FAISS"""
    resume_text = await file.read()
    resume_embedding = embedding_model.encode(resume_text.decode("utf-8")).astype(np.float32)

    # Search in FAISS
    _, indices = faiss_index.search(np.array([resume_embedding]), k=3)
    matched_jobs = [job_titles[i] for i in indices[0]]

    return {"matched_jobs": matched_jobs}

# 3️⃣ **Upload HR Policies & Interview FAQs**
@app.post("/upload/")
async def upload_files(hr_policies: UploadFile = None, interview_faqs: UploadFile = None):
    """Uploads HR policies and Interview FAQs"""
    if hr_policies:
        with open(HR_POLICIES_FILE, "wb") as f:
            f.write(await hr_policies.read())
    if interview_faqs:
        with open(INTERVIEW_FAQS_FILE, "wb") as f:
            f.write(await interview_faqs.read())
    return {"message": "Files uploaded successfully"}

# 4️⃣ **HR Chatbot Query**
class TextData(BaseModel):
    content: str

@app.post("/query/")
def chatbot_query(query: TextData):
    """Processes HR-related chatbot queries"""
    response = llm_model(query.content, max_length=200)[0]["generated_text"]
    return {"response": response}

# -------------------------------
# Run FastAPI
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
