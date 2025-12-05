import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
import os
import uuid

# ------------------------------------------
# CONFIG
# ------------------------------------------
genai.configure(api_key="AIzaSyDeDGPceej_95zbNO4u6qu9dfe7H2eNRL4")
model = genai.GenerativeModel("models/gemini-2.5-pro")

st.set_page_config(page_title="AI PDF Assistant", layout="wide")

# ------------------------------------------
# HEADER
# ------------------------------------------
st.markdown("""
<h1 style='text-align:center; color:#4B0082;'>📚 AI PDF Assistant</h1>
<h3 style='text-align:center; color:gray;'>Ask Anything From Your PDFs</h3>
<br>
""", unsafe_allow_html=True)

# ------------------------------------------
# SESSION STATE
# ------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "db" not in st.session_state:
    st.session_state.db = None

if "pdf_ids" not in st.session_state:
    st.session_state.pdf_ids = set()

VECTORSTORE_DIR = "faiss_store"
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ------------------------------------------
# SIDEBAR – MULTI PDF UPLOAD
# ------------------------------------------
st.sidebar.header("📁 Upload PDFs")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)

    for pdf in uploaded_files:
        pdf_id = pdf.name
        if pdf_id in st.session_state.pdf_ids:
            continue

        temp_path = f"temp_{uuid.uuid4()}.pdf"
        with open(temp_path, "wb") as f:
            f.write(pdf.getbuffer())

        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # Create or add to FAISS index
        if st.session_state.db is None:
            st.session_state.db = FAISS.from_documents(docs, emb)
        else:
            st.session_state.db.add_documents(docs)

        # Save persistent index
        st.session_state.db.save_local(VECTORSTORE_DIR)
        st.session_state.pdf_ids.add(pdf_id)

        os.remove(temp_path)

    st.sidebar.success("PDFs indexed successfully!")

# Load FAISS index if exists
elif os.path.exists(os.path.join(VECTORSTORE_DIR, "index.faiss")) and st.session_state.db is None:
    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    st.session_state.db = FAISS.load_local(
        VECTORSTORE_DIR, emb, allow_dangerous_deserialization=True
    )

# ------------------------------------------
# CHAT DISPLAY
# ------------------------------------------
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        color = "#DCF8C6" if role == "user" else "#E6E6FA"
        align = "flex-end" if role == "user" else "flex-start"

        st.markdown(
            f"""
            <div style='display:flex; justify-content:{align}; margin:8px 0;'>
                <div style='background:{color}; padding:12px; border-radius:10px; max-width:70%; word-wrap: break-word;'>
                    {content}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ------------------------------------------
# CHAT INPUT
# ------------------------------------------
query = st.chat_input("Ask about your PDFs...")

if query:
    if st.session_state.db is None:
        st.warning("⚠ Upload PDFs first!")
    else:
        # Show user message instantly
        st.session_state.messages.append({"role": "user", "content": query})
        st.rerun()  # Rerender to display question

# ------------------------------------------
# PROCESS LAST UNANSWERED USER MESSAGE
# ------------------------------------------
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_query = st.session_state.messages[-1]["content"]

    with st.spinner("Processing your question... ⏳"):
        # Optimized retrieval
        retriever = st.session_state.db.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(last_query)
        context = "\n".join([d.page_content for d in docs])

        # Prompt formatting with clear instruction
        prompt = f"""
        Use ONLY the context below to answer the question accurately.
        
        Context:
        {context}
        
        Question: {last_query}
        
        Answer:
        """

        response = model.generate_content(prompt)
        answer = response.text.strip()

    # Add assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
