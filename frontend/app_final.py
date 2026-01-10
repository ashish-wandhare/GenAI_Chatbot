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

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
st.sidebar.title(" More Links")

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

    doc_path = "data/raw/Python Programming.pdf"

    if os.path.exists(doc_path):
        with open(doc_path, "rb") as f:
            st.download_button(
                label="⬇️ Download Document",
                data=f,
                file_name="Python Programming.pdf",
                mime="application/pdf"
            )

        st.markdown("### Preview")
        with open(doc_path, "rb") as f:
            st.pdf(f.read())

# =====================================================
# ABOUT PAGE
# =====================================================
elif page == "About":
    st.title("⬜ About This Project")

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

    st.markdown("### ⬜  Developed By")
    st.write("Ashish W.")

    st.markdown("### ⬜  Use Cases")
    st.write("- Book-based Q&A\n- Internal company documents chatbot\n- Notes assistant")


# =====================================================
# CHATBOT PAGE
# =====================================================
else:
    # ✅ Sticky Heading
    st.markdown(
    """
    <style>
    /* Sidebar width default in Streamlit ≈ 21rem */
    :root{
        --sidebar-width: 21rem;
    }

    /* Give space for fixed header + fixed footer */
    .block-container {
    padding-top: 181px !important;  /* ✅ enough space so 1st message won't hide */
    padding-bottom: 71px !important;
    }

    /* ✅ FIXED Header (will NOT scroll) */
    .fixed-header {
    position: fixed;
    top: 3.2rem;               /* ✅ Push header down */
    left: var(--sidebar-width);
    width: calc(100% - var(--sidebar-width));
    z-index: 9999;
    background: white;
    padding: 14px 20px 10px 20px;   /* ✅ more padding */
    border-bottom: 1px solid #e6e6e6;
    }

    /* ✅ If sidebar collapsed (mobile), header takes full width */
    @media (max-width: 768px) {
        .fixed-header {
            left: 0;
            width: 100%;
        }
    }

    /* ✅ Chat bubbles */
    .answer-box {
        background: #f6fff8;
        border: 1px solid #b7eb8f;
        padding: 14px 16px;
        border-radius: 14px;
        font-size: 16px;
        line-height: 1.5;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
        margin-top: 5px;
    }

    .user-box {
        background: #f0f7ff;
        border: 1px solid #91caff;
        padding: 12px 14px;
        border-radius: 14px;
        font-size: 16px;
        line-height: 1.5;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.05);
        margin-top: 5px;
    }
    </style>

    <div class="fixed-header">
        <h1 style="margin:0;color:gray;">🤖 ChatGTP</h1>
        <h4 style="margin:0;color:gray;">GenAI Context-Aware Chatbot</h4>
    </div>
    """,
    unsafe_allow_html=True
)
    st.markdown("<div style='height:25px;'></div>", unsafe_allow_html=True)


    # Initialize chat memory
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    #_______________________________________________
    # Load RAG backend (cached)
    #_______________________________________________
    @st.cache_resource
    def load_rag_pipeline():
        embedding_model = HuggingFaceEmbeddings(
            model_name=config["embedding"]["model_name"]
        )

        # ✅ Local vs Cloud DB location
        if os.environ.get("STREAMLIT_SERVER_HEADLESS") == "true":
            persist_dir = os.path.join("/tmp", "chroma_db")
        else:
            persist_dir = config["vector_store"]["persist_directory"]

        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )

        # ✅ Create embeddings if DB is empty
        try:
            empty_db = vectordb._collection.count() == 0
        except Exception:
            empty_db = True

        if empty_db:
            st.info("📌 First run: Creating embeddings from PDF...")

            pdf_path = "data/raw/Python Programming.pdf"
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=150
            )
            chunks = splitter.split_documents(docs)

            vectordb.add_documents(chunks)
            vectordb.persist()

            st.success("✅ Embeddings created successfully!")

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
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        return qa_chain

    qa_chain = load_rag_pipeline()
    
    # Display chat history with UI
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):

            # Assistant Bubble
            if msg["role"] == "assistant":
                st.markdown(
                    f"""
                    <div class="answer-box">
                        <div class="answer-title"></div>
                        <div>{msg['content']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # User Bubble
            else:
                st.markdown(
                    f"""
                    <div class="user-box">
                        {msg['content']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # User input
    user_query = st.chat_input("Ask python questions from the documents...")

    if user_query:
        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Show user message
        with st.chat_message("user"):
            st.markdown(
                f"""
                <div class="user-box">
                    {user_query}
                </div>
                """,
                unsafe_allow_html=True
            )

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa_chain.invoke(user_query)

                st.markdown(
                    f"""
                    <div class="answer-box">
                        <div class="answer-title"></div>
                        <div>{response}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 6px 0;
        font-size: 12px;
        color: #777;
        background: white;
        border-top: 1px solid #e6e6e6;
        z-index: 999;
    }
    </style>

    <div class="footer">
        Developed by: <b>Ashish W.</b> • ChatGTP can make mistakes. Check important info
    </div>
    """,
    unsafe_allow_html=True
)

