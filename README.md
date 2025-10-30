# 🧠 Contract Intelligence API

An **AI-powered contract analysis API** built with **FastAPI**, **LangChain**, **OpenAI**, and **Pinecone**.  
It ingests PDF contracts, extracts key clauses, performs Q&A, and audits legal risks.

---

## 📁 Project Structure

app/

    ├── main.py # FastAPI entry point
    ├── config.py # Environment configuration
    ├── model.py # Pydantic data models
    ├── init.py # Service initialization
    └── services/

        ├── document_service.py # Handles PDF ingestion and Pinecone storage
        ├── qa_service.py # Question answering and data extraction
        └── audit_service.py # Contract risk auditing using LLMs

---

## ⚙️ Features

- **PDF Ingestion** → Parses and chunks contract documents.  
- **Vector Storage** → Stores embeddings in Pinecone with per-document namespaces.  
- **Structured Extraction** → Extracts key contract metadata (parties, term, governing law, etc.).  
- **Contract Q&A** → Answers user questions contextually from the uploaded contract.  
- **Audit Analysis** → Detects risks and generates structured JSON audit reports.  
- **Streaming Responses** → Supports live word-by-word streaming answers.

---

## 🧩 Tech Stack

- **Framework**: FastAPI  
- **LLM**: OpenAI GPT-4o-mini (via LangChain)  
- **Vector DB**: Pinecone  
- **Embeddings**: OpenAIEmbeddings  
- **Document Parsing**: PyPDFLoader (LangChain community)  
- **Chunking**: RecursiveCharacterTextSplitter  
- **Environment**: dotenv  

---
