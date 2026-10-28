# GenAI Context-Aware Chatbot
### Retrieval-Augmented Generation (RAG) with LangChain, Hugging Face, and ChromaDB

<p align="center">
  <img src="https://img.shields.io/badge/Generative%20AI-RAG-blue" alt="Generative AI">
  <img src="https://img.shields.io/badge/LLM-FLAN--T5-purple" alt="FLAN-T5">
  <img src="https://img.shields.io/badge/Framework-LangChain-green" alt="LangChain">
  <img src="https://img.shields.io/badge/Interface-Streamlit-red" alt="Streamlit">
</p>

## Overview

**GenAI Context-Aware Chatbot** is a Retrieval-Augmented Generation (RAG) application that answers questions using information retrieved from a provided PDF document.

The project combines document loading, text chunking, semantic embeddings, vector search, and a language model to generate responses grounded in the supplied document. The initial knowledge source is a Python book PDF.

Rather than relying only on the language model's pre-trained knowledge, the RAG workflow retrieves relevant document passages and supplies them as context for answer generation.

## Problem Statement

General-purpose language models can produce answers that are not supported by a specific reference document. When users need information from a particular book or PDF, they need a way to retrieve relevant content and use it as context for their questions.

This project demonstrates a document-grounded chatbot that uses a PDF as its knowledge source.

## Solution

The application uses a RAG pipeline to:

1. Load the source PDF.
2. Split its content into smaller text chunks.
3. Convert the chunks into vector embeddings.
4. Store the embeddings in a Chroma vector database.
5. Retrieve relevant chunks in response to a user question.
6. Pass the retrieved context and question to the FLAN-T5 language model.
7. Generate a response based on the retrieved context.

## Key Features

- **PDF-Based Knowledge:** Uses a provided Python book PDF as the chatbot's source document.
- **Semantic Retrieval:** Retrieves relevant document chunks using vector similarity.
- **Context-Aware Generation:** Supplies retrieved text to the language model as context for answering.
- **Local Vector Store:** Uses Chroma for storing and retrieving document embeddings.
- **Interactive Chat Interface:** Provides a Streamlit frontend for asking questions.
- **Separated Workflow:** Keeps document processing and embedding creation separate from chatbot inference.

## RAG Architecture

```text
                 PDF Document
                      |
                      v
             Document Loading
               (PyPDFLoader)
                      |
                      v
              Text Chunking
          (size: 500, overlap: 100)
                      |
                      v
              Text Embeddings
        (all-MiniLM-L6-v2)
                      |
                      v
              Chroma Vector DB
                      |
                      |      User Question
                      |            |
                      v            v
               Relevant Chunk Retrieval
                      |
                      v
            Question + Retrieved Context
                      |
                      v
              Hugging Face LLM
              (FLAN-T5-base)
                      |
                      v
                Final Answer
```

## Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Programming Language | Python | Application and pipeline development |
| Frontend | Streamlit | Chat interface |
| LLM | `google/flan-t5-base` | Context-aware response generation |
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` | Text embedding generation |
| Vector Store | ChromaDB | Persistent vector storage and retrieval |
| Orchestration | LangChain | Document and RAG pipeline components |
| Document Loader | PyPDFLoader | PDF text extraction |

## Configuration

The project configuration includes the following embedding and vector-store settings:

| Setting | Value |
|---|---|
| Embedding Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Chunk Size | `500` |
| Chunk Overlap | `100` |
| Vector Store | Chroma |
| Persistence Directory | `chroma_db` |
| Source Data Directory | `data` |

Chunk size and overlap control how the source document is divided into smaller passages before embedding and retrieval.

## Project Structure

The repository is organized into separate areas for the chatbot, configuration, source data, embeddings, and frontend.

```text
GenAI_Chatbot/
├── chatbot/          # Chatbot and RAG-related components
├── config/           # Model and pipeline configuration
├── data/
│   └── raw/         # Source document data
├── embeddings/       # Embedding and vector-store components
├── frontend/         # Streamlit application
├── requirements.txt   # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

- Python
- Git
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/ashish-wandhare/GenAI_Chatbot.git
```

### 2. Navigate to the Project Directory

```bash
cd GenAI_Chatbot
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Source PDF

Place the source PDF in the data directory expected by the project configuration. The original project uses a Python book PDF as its knowledge source.

### 5. Prepare the Embeddings / Vector Store

Run the project's document-ingestion and embedding creation workflow to process the PDF and populate the persistent Chroma vector store.

### 6. Launch the Chatbot

Run the Streamlit frontend entry point from the repository. The frontend application file is located in the `frontend/` directory.

> **Note:** Use the exact embedding-generation and Streamlit launch commands defined by the current project files. They may differ depending on the selected entry-point script and configuration.

## Usage

1. Start the chatbot application.
2. Enter a question related to the content of the provided PDF.
3. The RAG pipeline retrieves relevant document chunks.
4. The language model receives the question and retrieved context.
5. The chatbot displays a generated response.

## Design Considerations

- **Retrieval before generation:** Relevant passages are retrieved before the LLM generates its response.
- **Document-grounded context:** The source PDF supplies the context used in the RAG workflow.
- **Persistent vector storage:** Chroma can persist document embeddings for subsequent retrieval.
- **Separate ingestion and inference:** Document preparation is kept distinct from the user-facing question-answering flow.

## Future Enhancements

- Display source references and PDF page numbers with answers.
- Support multiple PDF documents.
- Experiment with alternative embedding models, such as MPNet.
- Add conversation memory for follow-up questions.
- Add optional cloud-hosted language models.
- Improve retrieval quality and answer evaluation.

## Repository

[**GenAI_Chatbot**](https://github.com/ashish-wandhare/GenAI_Chatbot)



AI/ML Engineer | Python | Machine Learning | Generative AI | RAG

