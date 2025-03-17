import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import FAISS_INDEX_PATH

# Load FAISS Index
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

# Load Transformer Model
model = SentenceTransformer("all-MiniLM-L6-v2")

def search_jobs(query, top_k=3):
    query_vector = model.encode([query]).astype('float32')
    distances, indices = faiss_index.search(query_vector, top_k)
    
    return indices[0], distances[0]

# Test Search
query = "Looking for a Python Developer with FastAPI experience"
results, scores = search_jobs(query)
print(f"Results: {results}, Scores: {scores}")
