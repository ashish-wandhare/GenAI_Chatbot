import yaml
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

from transformers import pipeline


# Load config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name=config["embedding"]["model_name"]
)

# Vector DB
vectordb = Chroma(
    persist_directory=config["vector_store"]["persist_directory"],
    embedding_function=embedding_model
)

retriever = vectordb.as_retriever(
    search_kwargs={"k": config["retriever"]["top_k"]}
)

# LOCAL LLM (CPU-safe, FREE)
hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=256
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

# RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)

# Test query
query = "What is Retrieval-Augmented Generation?"
response = qa_chain.run(query)

print("\n🔹 Question:")
print(query)
print("\n🔹 Answer:")
print(response)
