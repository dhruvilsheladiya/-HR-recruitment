from transformers import pipeline
from src.rag import retrieve_relevant_docs

# Load LLM (Mistral, Falcon, or LLaMA)
llm = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")

def generate_response(query: str):
    """Retrieve relevant HR info and generate structured response"""
    relevant_docs = retrieve_relevant_docs(query)
    context = " ".join(relevant_docs)

    prompt = f"HR Assistant: Based on our HR policies and job database, here’s the response:\n\n{context}\n\nQ: {query}\nA: "
    response = llm(prompt, max_length=300)
    
    return response[0]["generated_text"]
