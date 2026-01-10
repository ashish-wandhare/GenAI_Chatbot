import streamlit as st
import yaml
import warnings
import os

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


# =====================================================
# Sidebar Navigation
# =====================================================
st.sidebar.title("🔵 Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Chatbot", "Document", "About"]
)


# =====================================================
# DOCUMENT PAGE
# =====================================================
if page == "Document":
    st.title("📄 Document Viewer")

    st.write("Here you can open the document used for your chatbot training.")

    # Example: if your pdf/doc is stored in project folder
    doc_path = "data/raw/Python Programming.pdf"   # <-- change this path to your document

    if os.path.exists(doc_path):
        with open(doc_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Document",
                data=f,
                file_name="python_book.pdf",
                mime="application/pdf"
            )

        # PDF preview (works in many browsers)
        st.markdown("### Preview")
        with open(doc_path, "rb") as f:
            st.pdf(f.read())

    else:
        st.warning(f"Document not found at: `{doc_path}`")
        st.info("Update the file path in code or upload document below.")

        uploaded_file = st.file_uploader("Upload Document", type=["pdf"])
        if uploaded_file:
            st.success("Document uploaded successfully!")
            st.download_button("Download Uploaded Document", uploaded_file)


# =====================================================
# ABOUT PAGE
# =====================================================
elif page == "About":
    st.title("About This Project")

    st.markdown("""
### 🤖 GenAI Context-Aware Chatbot (RAG)

This project is a **Retrieval-Augmented Generation (RAG)** chatbot built using:

- **Streamlit** (UI)
- **HuggingFace Embeddings**
- **Chroma Vector Store**
- **LangChain**
- **FLAN-T5 model**

It answers questions ONLY from the provided documents.  
If the answer is not found in docs, it responds:  
> "I don't know based on the provided documents."
""")

    st.markdown("### Developed By")
    st.write("Ashish W.")

    st.markdown("### Use Cases")
    st.write("- Book-based Q&A\n- Internal company documents chatbot\n- Notes assistant")


# =====================================================
# CHATBOT PAGE (your existing code)
# =====================================================
else:
    st.markdown(
    """
    <style>
    /* Make the header sticky within main content area */
    div[data-testid="stVerticalBlock"] > div:has(.main-sticky-header) {
        position: sticky;
        top: 3.5rem;              /* below Streamlit top bar */
        z-index: 999;
        background: white;
        padding-top: 8px;
        padding-bottom: 8px;
        border-bottom: 1px solid #e6e6e6;
    }

    /* Add some spacing below the header */
    .main-sticky-header {
        padding: 8px 0px;
    }
    </style>

    <div class="main-sticky-header">
        <h2 style="margin:0;">🤖 GenAI Context-Aware Chatbot</h2>
        <p style="margin:0; color: gray;">Ask questions based on Python Book documents</p>
    </div>
    """,
    unsafe_allow_html=True
)

    # Initialize chat memory
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Load RAG backend (cached)
    @st.cache_resource
    def load_rag_pipeline():
        embedding_model = HuggingFaceEmbeddings(
            model_name=config["embedding"]["model_name"]
        )

        vectordb = Chroma(
            persist_directory=config["vector_store"]["persist_directory"],
            embedding_function=embedding_model
        )

        retriever = vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6, "fetch_k": 20}
        )

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

        hf_pipeline = pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_new_tokens=256
        )

        llm = HuggingFacePipeline(pipeline=hf_pipeline)

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

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                st.markdown(f"✅ **Answer:** {msg['content']}")
            else:
                st.markdown(msg["content"])
            

    # User input
    user_query = st.chat_input("Ask something about your documents...")

    if user_query:
        st.session_state.messages.append(
            {"role": "user", "content": user_query}
        )
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa_chain.invoke(user_query)
                #st.markdown(response)
                st.markdown(f"✅ **Answer:** {response}")

        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )
