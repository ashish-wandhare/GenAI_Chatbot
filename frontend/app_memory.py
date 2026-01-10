import streamlit as st
import yaml
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from transformers import pipeline

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline


# =====================================================
# Helper: format chat history (SHORT-TERM MEMORY)
# =====================================================
def format_chat_history(messages, max_turns=5):
    """
    Keeps only the last N turns (user + assistant).
    This prevents long-context pollution.
    """
    recent = messages[-max_turns * 2:]
    history = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        history.append(f"{role}: {msg['content']}")
    return "\n".join(history)


# =====================================================
# Helper: format retrieved documents
# =====================================================
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# =====================================================
# Load configuration
# =====================================================
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)


# =====================================================
# Streamlit page config
# =====================================================
st.set_page_config(
    page_title="GenAI Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 GenAI Context-Aware Chatbot")
st.write("Ask questions based on Python Book documents")


# =====================================================
# Initialize chat memory (MUST come before pipeline load)
# =====================================================
if "messages" not in st.session_state:
    st.session_state.messages = []


# =====================================================
# Load RAG backend (cached)
# =====================================================
@st.cache_resource
def load_rag_pipeline():

    # Embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name=config["embedding"]["model_name"]
    )

    # Vector DB
    vectordb = Chroma(
        persist_directory=config["vector_store"]["persist_directory"],
        embedding_function=embedding_model
    )

    # Retriever (MMR for better diversity)
    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20}
    )

    # Prompt with memory
    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template="""
You are an expert assistant.

Use ONLY the information provided in the context below.
If the answer is not explicitly stated, say:
"I don't know based on the provided documents."

Conversation so far:
{chat_history}

Context:
{context}

Question:
{question}

Answer clearly and concisely:
"""
    )

    # Local LLM
    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # RAG chain with short-term memory
    qa_chain = (
        {
            "context": retriever | format_docs,
            "chat_history": lambda _: format_chat_history(
                st.session_state.get("messages", [])
            ),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return qa_chain


qa_chain = load_rag_pipeline()


# =====================================================
# Display chat history
# =====================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =====================================================
# User input
# =====================================================
user_query = st.chat_input("Ask something about your documents...")

if user_query:
    # User message
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )
    with st.chat_message("user"):
        st.markdown(user_query)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = qa_chain.invoke(user_query)
            st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
