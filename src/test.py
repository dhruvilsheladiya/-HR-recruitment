import pickle

try:
    with open("embeddings/docs.pkl", "rb") as f:
        job_listings = pickle.load(f)
    print("✅ docs.pkl loaded successfully!")
    print("Job Listings:", job_listings)
except Exception as e:
    print(f"❌ Error loading docs.pkl: {e}")
