import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import fitz  # PyMuPDF
from PIL import Image
import io
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import subprocess

subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])




load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Load NLP model for keyword extraction
nlp = spacy.load("en_core_web_sm")

# Set page config at the very beginning
st.set_page_config(
    page_title="Chat with Research Paper",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

working_dir = os.path.dirname(os.path.abspath(__file__))

# Sidebar with File Upload, Chunking Settings, and Document Controls
st.sidebar.title("Document Controls")
uploaded_file = st.sidebar.file_uploader(label="Upload Your PDF file", type=["pdf"])

chunk_size = st.sidebar.slider(
    label="Chunk Size for Text Splitting",
    min_value=500, 
    max_value=2000, 
    value=1000,
    step=100
)
chunk_overlap = st.sidebar.slider(
    label="Chunk Overlap for Text Splitting",
    min_value=100, 
    max_value=500, 
    value=200,
    step=50
)

# Load document 
def load_document(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# Chunking and storing in the vector database
def setup_vectorstore(documents, chunk_size, chunk_overlap):
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore

# Chaining
def create_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0,
        api_key=groq_api_key
    )
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )
    return chain

# Summarize document
def summarize_document(documents):
    full_text = " ".join([doc.page_content for doc in documents])
    summary_prompt = f"Please provide a concise summary of the following text:\n\n{full_text[:10000]}" 
    response = st.session_state.conversation_chain({"question": summary_prompt})
    return response["answer"]

# Keyword extraction using spaCy
def extract_keywords(text, top_n=10):
    doc = nlp(text)
    keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
    tfidf = TfidfVectorizer(max_features=top_n)
    tfidf_matrix = tfidf.fit_transform([text])
    return tfidf.get_feature_names_out()

# Advanced search functionality
def search_document(text, query):
    results = []
    for i, page in enumerate(text):
        if query.lower() in page.lower():
            snippet = page[:min(300, len(page))]
            results.append((i + 1, snippet))
    return results

st.title(" 🦙 Chat With Document 📄")

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None
if "documents" not in st.session_state:
    st.session_state.documents = None
if "keywords" not in st.session_state:
    st.session_state.keywords = None

# Process the uploaded PDF file
if uploaded_file and not st.session_state.file_processed:
    with st.spinner("Processing document... This may take a moment."):
        file_path = os.path.join(working_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            st.session_state.documents = load_document(file_path)
            st.session_state.vectorstore = setup_vectorstore(st.session_state.documents, chunk_size, chunk_overlap)
            st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)
            st.session_state.file_processed = True
            st.session_state.pdf_path = file_path
            st.success("Document processed successfully!")
        except Exception as e:
            st.error(f"An error occurred while processing the document: {str(e)}")

# Create Tabs for Document Interaction
if st.session_state.file_processed:
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Summary", "💬 Chat", "👁️ PDF Preview", "🔑 Keywords & Search"])

    # Tab 1: Summary
    with tab1:
        if st.button("Summarize Document"):
            with st.spinner("Generating summary..."):
                if st.session_state.conversation_chain:
                    summary = summarize_document(st.session_state.documents)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": f"Document Summary:\n{summary}", 
                        "timestamp": timestamp
                    })
                    st.write(summary)
                else:
                    st.error("Please wait for the document to be fully processed before summarizing.")

    # Tab 2: Chat
    with tab2:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(f"**{message['timestamp']}**")
                st.markdown(message["content"])

        user_input = st.chat_input("Ask about the document")
        if user_input:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.chat_history.append({"role": "user", "content": user_input, "timestamp": timestamp})
            
            with st.chat_message("user"):
                st.markdown(f"**{timestamp}**")
                st.markdown(user_input)
            
            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    response = st.session_state.conversation_chain({"question": user_input})
                assistant_response = response["answer"]
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.markdown(f"**{timestamp}**")
                st.markdown(assistant_response)
                st.session_state.chat_history.append({"role": "assistant", "content": assistant_response, "timestamp": timestamp})

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.experimental_rerun()

    # Tab 3: PDF Preview
    with tab3:
        if st.session_state.pdf_path:
            with open(st.session_state.pdf_path, "rb") as pdf_file:
                PDFbyte = pdf_file.read()

            st.download_button(label="Download PDF", 
                               data=PDFbyte,
                               file_name="document.pdf",
                               mime='application/octet-stream')
            
            pdf_document = fitz.open(st.session_state.pdf_path)
            num_pages = len(pdf_document)

            st.write(f"PDF Preview (Total Pages: {num_pages}):")

            page_number = st.number_input("Enter page number", min_value=1, max_value=num_pages, value=1)

            page = pdf_document[page_number - 1]

            # Render page to an image
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()

            # Display the image
            st.image(img_byte_arr, use_column_width=True)

            pdf_document.close()

    # Tab 4: Keywords & Search
    with tab4:
        full_text = " ".join([doc.page_content for doc in st.session_state.documents])
        
        if st.button("Extract Keywords"):
            keywords = extract_keywords(full_text)
            st.session_state.keywords = keywords
            st.write("Extracted Keywords:", ", ".join(keywords))
        
        # Advanced search bar
        search_query = st.text_input("Search in Document")
        if search_query:
            search_results = search_document(full_text.split('\n'), search_query)
            if search_results:
                st.write(f"Found {len(search_results)} results for '{search_query}':")
                for page_num, snippet in search_results:
                    st.write(f"Page {page_num}: {snippet}...")
            else:
                st.write("No results found.")
