import pandas as pd
import fitz  # PyMuPDF
import faiss
import spacy
import chromadb
from sentence_transformers import SentenceTransformer
import os
from config import DATA_PATH, FAISS_INDEX_PATH, CHROMA_DB_PATH

# Load NLP Model for NER
nlp = spacy.load("en_core_web_sm")

# Load Sentence Transformer
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load Job Listings from CSV
csv_file = os.path.join(DATA_PATH, "job_listings.csv")
df = pd.read_csv(csv_file)

# Extract job descriptions
job_descriptions = df["job_description"].tolist()

# Extract Entities (NER)
def extract_entities(text):
    doc = nlp(text)
    return {ent.label_: ent.text for ent in doc.ents}

# Store in ChromaDB
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection("job_listings")

for idx, job in enumerate(job_descriptions):
    collection.add(
        ids=[str(idx)],
        metadatas=[{"job_id": idx, "entities": extract_entities(job)}],
        documents=[job],
    )

# Encode & Store in FAISS
job_vectors = model.encode(job_descriptions).astype('float32')
faiss_index = faiss.IndexFlatL2(job_vectors.shape[1])
faiss_index.add(job_vectors)

faiss.write_index(faiss_index, FAISS_INDEX_PATH)
print("✅ FAISS & ChromaDB Index Created!")
