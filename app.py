import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import google.generativeai as genai
import os
import uuid
from pdf2image import convert_from_bytes
import pytesseract
from langchain_community.docstore.document import Document

# ------------------------------------------
# CONFIG
# ------------------------------------------
api_key = st.secrets["google"]["api_key"]  # Safe API key
genai.configure(api_key=api_key)
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
# SIDEBAR – MOBILE-FRIENDLY PDF UPLOAD
# ------------------------------------------
st.sidebar.header("📁 Upload PDFs")
st.sidebar.markdown("📌 **Tip:** On mobile, tap 'Choose Files' to select PDFs from your device.")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs", type=["pdf"], accept_multiple_files=True, key="pdf_uploader", label_visibility="visible"
)

if uploaded_files:
    # Clear previous session data to avoid old PDFs persisting
    st.session_state.db = None
    st.session_state.pdf_ids = set()
    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)

    for pdf in uploaded_files:
        pdf_id = pdf.name
        if pdf_id in st.session_state.pdf_ids:
            continue

        temp_path = f"temp_{uuid.uuid4()}.pdf"
        with open(temp_path, "wb") as f:
            f.write(pdf.getbuffer())

        # Load PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # If PDF has no extractable text, use OCR for scanned/handwritten pages
        if not docs:
            pages = convert_from_bytes(open(temp_path, "rb").read())
            docs = []
            for page in pages:
                text = pytesseract.image_to_string(page, lang='eng', config='--oem 1 --psm 6')
                if text.strip():
                    docs.append(Document(page_content=text))

        # Create or add to FAISS index
        if st.session_state.db is None:
            st.session_state.db = FAISS.from_documents(docs, emb)
        else:
            st.session_state.db.add_documents(docs)

        st.session_state.pdf_ids.add(pdf_id)
        os.remove(temp_path)

    # Save persistent index
    st.session_state.db.save_local(VECTORSTORE_DIR)
    st.sidebar.success("✅ PDFs indexed successfully!")

# Load FAISS index if exists and session DB is empty
elif os.path.exists(os.path.join(VECTORSTORE_DIR, "index.faiss")) and st.session_state.db is None:
    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    st.session_state.db = FAISS.load_local(
        VECTORSTORE_DIR, emb, allow_dangerous_deserialization=True
    )

# ------------------------------------------
# CHAT DISPLAY – USER RIGHT, ASSISTANT LEFT
# ------------------------------------------
st.markdown("""
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.chat-row {
    display: flex;
    width: 100%;
}

.chat-bubble {
    padding: 12px;
    border-radius: 10px;
    max-width: 70%;
    word-wrap: break-word;
    font-size: 16px;
}

/* User on right */
.user {
    background-color: #DCF8C6;
    color: black;
    margin-left: auto;
}

/* Assistant on left */
.assistant {
    background-color: #E6E6FA;
    color: black;
    margin-right: auto;
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
    .user {
        background-color: #2e7d32;
        color: #EEE;
    }
    .assistant {
        background-color: #444;
        color: #EEE;
    }
}

/* Mobile adjustments */
@media (max-width: 600px) {
    .chat-bubble {
        max-width: 90%;
        font-size: 14px;
    }
    .chat-row {
        flex-direction: column;
        align-items: flex-start;
    }
    .assistant {
        align-self: flex-start;
    }
}
</style>
""", unsafe_allow_html=True)

# Chat container
chat_container = st.container()
with chat_container:
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        st.markdown(
            f"<div class='chat-row'><div class='chat-bubble {role}'>{content}</div></div>",
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

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
        retriever = st.session_state.db.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(last_query)
        context = "\n".join([d.page_content for d in docs])

        prompt = f"""
        Use ONLY the context below to answer the question accurately.
        
        Context:
        {context}
        
        Question: {last_query}
        
        Answer:
        """

        response = model.generate_content(prompt)
        answer = response.text.strip()

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()

