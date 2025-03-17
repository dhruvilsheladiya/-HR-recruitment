# HR Recruitment API

## 📌 Overview
This is an AI-powered HR recruitment API that assists candidates with job-related inquiries, matches resumes with job listings, and retrieves HR policies using Retrieval-Augmented Generation (RAG). It utilizes FAISS for job matching, ChromaDB for document retrieval, and an open-source LLM (Mistral-7B) for natural language processing.

---
## 🚀 Features
- **Job Listing Management** – Add and retrieve job descriptions.
- **Resume Matching** – Match candidates' resumes with job listings using FAISS.
- **HR Chatbot** – Answer HR-related queries using an AI model.
- **RAG-Based Query System** – Retrieve relevant HR documents for precise answers.
- **File Upload Support** – Upload HR policies and interview FAQs.


---
## 🛠️ Setup Instructions
### 1️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 2️⃣ Run FastAPI Server
```sh
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
🔗 **Access API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---
## 📡 API Endpoints
### ✅ 1. **Add Job Listing**
- **Endpoint:** `POST /add_job/`
- **Request Body:**
  ```json
  {
    "title": "Software Engineer",
    "description": "Develop web applications using Python and React."
  }
  ```
- **Response:**
  ```json
  {"message": "Job added successfully"}
  ```

### ✅ 2. **Upload Resume & Match Jobs**
- **Endpoint:** `POST /match_jobs/`
- **Request Body:** File Upload (PDF or Text)
- **Response:**
  ```json
  {"matched_jobs": ["Software Engineer", "Backend Developer"]}
  ```

### ✅ 3. **Upload HR Policies & Interview FAQs**
- **Endpoint:** `POST /upload/`
- **Request Body:** File Upload (HR policies, interview FAQs)
- **Response:**
  ```json
  {"message": "Files uploaded successfully"}
  ```

### ✅ 4. **HR Chatbot Query**
- **Endpoint:** `POST /query/`
- **Request Body:**
  ```json
  {"content": "What is the leave policy?"}
  ```
- **Response:**
  ```json
  {"response": "Employees are entitled to 20 days of leave per year."}
  ```

### ✅ 5. **RAG-Based Query with ChromaDB**
- **Endpoint:** `POST /query_rag/`
- **Request Body:**
  ```json
  {"content": "Explain the company's hiring process."}
  ```
- **Response:**
  ```json
  {"response": "The hiring process includes screening, interviews, and onboarding."}


-- Its all about show in automatically in src/uploads folders and src/data folders.

---
## 📦 Deployment with Docker
### 1️⃣ Build the Docker Image
```sh
docker build -t hr_recruitment_api .
```
### 2️⃣ Run the Container
```sh
docker run -p 8000:8000 hr_recruitment_api
```

---
## 📌 Future Enhancements
✅ Add authentication and role-based access control (RBAC).  
✅ Improve job-matching algorithm with deep learning models.  
✅ Expand HR chatbot with additional legal and policy documents.  

---
## 📞 Contact
For support, please contact **Dhruvil Sheladiya**.

