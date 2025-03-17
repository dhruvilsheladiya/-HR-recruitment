import faiss
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer

# Ensure embeddings directory exists
os.makedirs("embeddings", exist_ok=True)

# Load Sentence Transformer Model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample job descriptions (replace with actual data)
job_listings = [
    "Software Engineer with experience in Python, FastAPI, and FAISS",
    "Data Scientist proficient in Machine Learning and NLP",
    "HR Manager with expertise in talent acquisition",
    "Backend Developer with cloud computing experience",
    "AI Researcher specializing in deep learning",
]

# Convert job descriptions into embeddings
job_vectors = model.encode(job_listings).astype('float32')

# Initialize FAISS index
dimension = job_vectors.shape[1]  # Get embedding size
index = faiss.IndexFlatL2(dimension)
index.add(job_vectors)

# Save the FAISS index
faiss.write_index(index, "embeddings/faiss_index.bin")

# Save job descriptions mapping
with open("embeddings/docs.pkl", "wb") as f:
    pickle.dump(job_listings, f)

print("✅ FAISS index and docs.pkl successfully created and saved!")
