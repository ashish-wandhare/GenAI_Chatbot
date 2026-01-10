import os
import yaml

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# -----------------------------
# Load config.yaml
# -----------------------------
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Paths
raw_data_path = config["paths"]["raw_data"]
persist_directory = config["vector_store"]["persist_directory"]

# -----------------------------
# Load TXT and PDF files
# -----------------------------
documents = []

for file in os.listdir(raw_data_path):
    file_path = os.path.join(raw_data_path, file)

    if file.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
        documents.extend(loader.load())

    elif file.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())

# -----------------------------
# Split text into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=config["embedding"]["chunk_size"],
    chunk_overlap=config["embedding"]["chunk_overlap"]
)

chunks = text_splitter.split_documents(documents)

# -----------------------------
# Load embedding model
# -----------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name=config["embedding"]["model_name"]
)

# -----------------------------
# Create Chroma DB
# -----------------------------
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=persist_directory
)

vectordb.persist()

print(" Embeddings created and stored in Chroma DB successfully (TXT + PDF).")
