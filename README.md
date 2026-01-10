📘 GenAI Context-Aware Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) chatbot built using LangChain, HuggingFace models, Chroma vector database, and Streamlit.
The chatbot answers questions only from a provided PDF (Python book), ensuring grounded and reliable responses.


📘 GenAI Context-Aware Chatbot (RAG)

A Retrieval-Augmented Generation (RAG) chatbot built using LangChain, HuggingFace models, Chroma vector database, and Streamlit. 
The chatbot answers questions only from a provided PDF (Python book), ensuring grounded and reliable responses.

User Question
     ↓
Vector Retriever (Chroma + Embeddings)
     ↓
Relevant PDF Chunks
     ↓
Prompt + Context
     ↓
HuggingFace LLM (FLAN-T5)
     ↓
Final Answer

| Component       | Technology                             |
| --------------- | -------------------------------------- |
| Frontend        | Streamlit                              |
| LLM             | HuggingFace (google/flan-t5-base)      |
| Embeddings      | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store    | Chroma                                 |
| Framework       | LangChain (v1.x)                       |
| Document Loader | PyPDFLoader                            |
| Language        | Python                                 |


GenAI_Chatbot/
│
├── frontend/
│   └── app.py                # Streamlit chatbot UI
│
├── config/
│   └── config.yaml           # Model & chunk configuration
│
├── data/
│   └── python_book.pdf       # Source PDF
│
├── chroma_db/                # Persistent vector database
│
├── create_embeddings.py      # PDF ingestion & embedding creation
│
├── requirements.txt
└── README.md

embedding:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  chunk_size: 500
  chunk_overlap: 100

vector_store:
  persist_directory: chroma_db

paths:
  raw_data: data


Practical GenAI system design

Clean separation of ingestion & inference

📚 Future Improvements

Add source citations (page numbers)

Support multiple PDFs

Improve embeddings (e.g., MPNet)

Add conversation memory

Enable cloud-based LLMs (optional)

👤 Author

Ashish Wandhare
Internship Project
Domain: Generative AI, LLM, RAG Systems