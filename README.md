# Context-Aware GenAI Chatbot — RAG + LangChain

A Retrieval-Augmented Generation (RAG) chatbot that answers questions using information retrieved from a provided PDF document. The project combines document embeddings, a Chroma vector database, a HuggingFace language model, and a Streamlit interface.


## 📌 Overview

Traditional chatbots may generate answers that are not grounded in a specific source. This project uses a RAG workflow to retrieve relevant passages from a PDF and provide them as context to the language model before generating a response.

The chatbot is designed to answer questions based on the content of the provided document.

## ✨ Key Features

- Context-aware question answering based on a source PDF
- PDF ingestion and text chunking
- Semantic document retrieval using HuggingFace embeddings
- Persistent vector storage with ChromaDB
- Response generation using FLAN-T5
- Interactive Streamlit chatbot interface
- Separate document-ingestion and chatbot-inference components

## 🔄 RAG Workflow

1. Load the source PDF.
2. Split the extracted text into chunks.
3. Convert chunks into embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
4. Store the embeddings in ChromaDB.
5. Retrieve relevant chunks for a user's question.
6. Pass the question and retrieved context to FLAN-T5.
7. Display the generated response in the Streamlit interface.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| LLM | HuggingFace `google/flan-t5-base` |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database | ChromaDB |
| RAG Framework | LangChain |
| PDF Processing | PyPDFLoader |

## 📂 Project Structure

```text
GenAI_Chatbot/
├── frontend/
│   └── app.py                 # Streamlit chatbot UI
├── config/
│   └── config.yaml            # Model and chunk configuration
├── data/
│   └── python_book.pdf        # Source PDF document
├── chroma_db/                 # Persistent vector database
├── create_embeddings.py        # PDF ingestion and embedding creation
├── requirements.txt             # Python dependencies
└── README.md
```

## ⚙️ Configuration

The project configuration includes:

- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Chunk size:** `500`
- **Chunk overlap:** `100`
- **Vector store directory:** `chroma_db`
- **Source data directory:** `data`

Configuration values are maintained in `config/config.yaml`.

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/ashish-wandhare/GenAI_Chatbot.git
cd GenAI_Chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

**Windows PowerShell:**

```powershell
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**

```bat
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create document embeddings

Make sure the source PDF is available in the `data/` directory, then run:

```bash
python create_embeddings.py
```

This processes the document and creates or updates the persistent vector database in `chroma_db/`.

Run this step again if you change the source document or relevant embedding configuration.

### 5. Launch the Streamlit application

```bash
streamlit run frontend/app.py
```

Open the local URL shown in the terminal.

By default, Streamlit is available at:

```text
http://localhost:8501
```

### 6. Ask questions

Enter a question related to the content of the source PDF in the chatbot interface.

The application retrieves relevant document chunks and uses them as context for generating a response.

## 🌐 Deployment

The application is also deployed on the following platforms:

- **Streamlit Cloud:** [Live Demo](https://ashish-genai-chatgtp.streamlit.app/)
- **AWS EC2:** [Live Demo](http://3.107.237.212:8502/)

## 🧠 Design Considerations

- Retrieval helps ground generated answers in the provided document.
- The embedding model and language model serve different roles: embeddings support semantic retrieval, while FLAN-T5 generates the response.
- The document-ingestion process is separated from the interactive chatbot interface.
- ChromaDB persistence avoids rebuilding the vector store on every application launch.

## 🔮 Future Enhancements

- Display source references and page numbers in responses
- Support multiple PDF documents
- Experiment with alternative embedding models, such as MPNet
- Add conversation memory
- Provide optional cloud-hosted LLM integrations

## 🔗 Project Links

- **GitHub:** [Repository](https://github.com/ashish-wandhare/GenAI_Chatbot)
- **Streamlit Cloud:** [Live Demo](https://ashish-genai-chatgtp.streamlit.app/)
- **AWS EC2:** [Live Demo](http://3.107.237.212:8502/)


[GitHub Profile](https://github.com/ashish-wandhare)

## 📸 Application Screenshots

<p align="center">
  <img width="1916" height="947" alt="Chatbot Main Interface" src="https://github.com/user-attachments/assets/3064eaf3-28be-41e7-bb03-032f0e187dbd" />
</p>

<p align="center">
  <img width="1917" height="942" alt="Document Interface" src="https://github.com/user-attachments/assets/7bbf4ea4-164b-438f-ace9-5bcc8d3c1f12" />
</p>

<p align="center">
  <img width="1912" height="947" alt="About Interface" src="https://github.com/user-attachments/assets/743ef1ee-65e5-466d-99bb-2972255fa424" />
</p>

<p align="center">
  <img width="1440" height="756" alt="GenAI Chatbot Snapshot" src="https://github.com/user-attachments/assets/e93b1af4-9c01-4aeb-b703-6e08ddd9d52f" />
</p>
