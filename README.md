# 📄 Chat with Research Paper 🦙

This Streamlit-based web application allows users to upload and interact with research papers in PDF format through conversational AI. The app leverages advanced language models and vector databases to provide functionalities like summarization, keyword extraction, and advanced search. It's designed to make research papers more accessible and easier to navigate by breaking the document into manageable chunks and enabling users to ask questions in a conversational manner.

## Key Features:

- **PDF Upload & Processing**: Upload PDF files of research papers for interactive exploration.
- **Text Chunking**: Split documents into smaller chunks using configurable chunk sizes and overlap settings for efficient text processing.
- **Conversational AI**: Chat with the research paper using LLaMA-based models powered by Groq. Ask questions about the document and receive insightful responses in real-time.
- **Document Summarization**: Generate concise summaries of uploaded research papers to quickly grasp the content.
- **Keyword Extraction**: Use NLP techniques to extract important keywords from the document for quick topic identification.
- **Advanced Search**: Search through the document using custom queries to find specific information or snippets of interest.
- **PDF Preview**: View and navigate through the uploaded PDF document, including the ability to download it.
- **Memory-Enabled Conversations**: Store and maintain chat history with the document using conversational memory to track your questions and responses.
- **Multi-Tab Interaction**: Explore the document through different views—summary, chat, PDF preview, and keyword search.

## How to Use:

1. **Upload a PDF**: Use the sidebar to upload a PDF document you want to explore.
2. **Configure Chunking**: Adjust chunk size and overlap settings to fine-tune text splitting.
3. **Interact with the Document**: Use the chat interface to ask questions, generate summaries, or extract keywords.
4. **Search and Preview**: Search for specific terms or preview individual pages of the document.
5. **Download PDF**: Download the uploaded PDF directly from the app.

## Requirements:

- Python 3.7+
- Streamlit
- LangChain
- HuggingFace Transformers
- FAISS
- PyMuPDF
- SpaCy (for NLP tasks)
- TfidfVectorizer (for keyword extraction)

## Installation:

```bash
git clone /https://github.com/faiquememon/chat_with_doc.git
cd chat-with-research-paper
pip install -r requirements.txt
streamlit run main.py
