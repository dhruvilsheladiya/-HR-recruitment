import faiss
import pickle
from sentence_transformers import SentenceTransformer
from src.config import FAISS_INDEX_PATH

# Load model & FAISS index
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
faiss_index = faiss.read_index(FAISS_INDEX_PATH)

# Load stored docs
with open("embeddings/docs.pkl", "rb") as f:
    docs = pickle.load(f)

def retrieve_relevant_docs(query: str, top_k=3):
    """Retrieve top K most relevant job descriptions or HR policies"""
    query_embedding = embed_model.encode([query])
    _, indices = faiss_index.search(query_embedding, top_k)
    return [docs[i] for i in indices[0]]
