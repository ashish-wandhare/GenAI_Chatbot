Context-Aware GenAI Chatbot — RAG + LangChain

A Retrieval-Augmented Generation (RAG) chatbot that answers questions using information retrieved from a provided PDF document. The project combines document embeddings, a Chroma vector database, a HuggingFace language model, and a Streamlit interface.

<p align="center">
  <img width="1916" height="947" alt="1_chatbot_main" src="https://github.com/user-attachments/assets/3064eaf3-28be-41e7-bb03-032f0e187dbd" />
  <img width="1917" height="942" alt="2_Document" src="https://github.com/user-attachments/assets/7bbf4ea4-164b-438f-ace9-5bcc8d3c1f12" />
  <img width="1912" height="947" alt="3_About" src="https://github.com/user-attachments/assets/743ef1ee-65e5-466d-99bb-2972255fa424" />
  <img width="1440" height="756" alt="GenAI_snap" src="https://github.com/user-attachments/assets/e93b1af4-9c01-4aeb-b703-6e08ddd9d52f" />

  /p>
  Overview

Traditional chatbots may generate answers that are not grounded in a specific source. This project uses a RAG workflow to retrieve relevant passages from a PDF and provide them as context to the language model before generating a response.

Key Features

Context-aware question answering based on a source PDF

PDF ingestion and text chunking

Semantic document retrieval using HuggingFace embeddings

Persistent vector storage with ChromaDB

Response generation using FLAN-T5

Interactive Streamlit chatbot interface

Separate document-ingestion and chatbot-inference components

RAG Workflow

Load the source PDF.

Split the extracted text into chunks.

Convert chunks into embeddings using sentence-transformers/all-MiniLM-L6-v2.

Store the embeddings in ChromaDB.

Retrieve relevant chunks for a user's question.

Pass the question and retrieved context to FLAN-T5.

Display the generated response in the Streamlit interface.

Technology Stack

Component

Technology

Programming language

Python

User interface

Streamlit

LLM

HuggingFace google/flan-t5-base

Embedding model

sentence-transformers/all-MiniLM-L6-v2

Vector database

ChromaDB

RAG framework

LangChain

PDF processing

PyPDFLoader

Project Structure

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

Configuration

The project configuration includes:

Embedding model: sentence-transformers/all-MiniLM-L6-v2

Chunk size: 500

Chunk overlap: 100

Vector store directory: chroma_db

Source data directory: data

Configuration values are maintained in config/config.yaml.

How to Run

1. Clone the repository

git clone https://github.com/ashish-wandhare/GenAI_Chatbot.git
cd GenAI_Chatbot

2. Create and activate a virtual environment

python -m venv .venv

On Windows PowerShell:

.venv\Scripts\Activate.ps1

On Windows Command Prompt:

.venv\Scripts\activate

On macOS or Linux:

source .venv/bin/activate

3. Install dependencies

pip install -r requirements.txt

4. Create document embeddings

Make sure the source PDF is available in the data/ directory, then run:

python create_embeddings.py

This processes the document and creates or updates the persistent vector database in chroma_db/. Run this step again if you change the source document or relevant embedding configuration.

5. Launch the Streamlit app

streamlit run frontend/app.py

Open the local URL shown in the terminal. By default, Streamlit is available at:

http://localhost:8501

6. Ask questions

Enter a question related to the content of the source PDF in the chatbot interface. The application retrieves relevant document chunks and uses them as context for generating a response.

Deployment

The application has also been deployed:

Streamlit Cloud: Live Demo

AWS EC2: Live Demo

Design Considerations

Retrieval helps ground generated answers in the provided document.

The embedding model and language model serve different roles: embeddings support semantic retrieval, while FLAN-T5 generates the response.

The document-ingestion process is separated from the interactive chatbot interface.

ChromaDB persistence avoids rebuilding the vector store on every application launch.

Future Enhancements

Display source references and page numbers in responses

Support multiple PDF documents

Experiment with alternative embedding models, such as MPNet

Add conversation memory

Provide optional cloud-hosted LLM integrations

Repository

GitHub — GenAI_Chatbot
