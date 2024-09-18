<h1>🦙 Chat With Research Paper 📄</h1>

<p style="font-size:18px;">This Streamlit-based web application allows users to upload and interact with research papers in PDF format through conversational AI. The app leverages advanced language models and vector databases to provide functionalities like summarization, keyword extraction, and advanced search.</p>

<h2>Key Features:</h2>
<ul>
    <li><strong>PDF Upload & Processing:</strong> <p style="font-size:16px;">Upload PDF files of research papers for interactive exploration.</p></li>
    <li><strong>Text Chunking:</strong> <p style="font-size:16px;">Split documents into smaller chunks using configurable chunk sizes and overlap settings for efficient text processing.</p></li>
    <li><strong>Conversational AI:</strong> <p style="font-size:16px;">Chat with the research paper using LLaMA-based models powered by Groq. Ask questions about the document and receive insightful responses in real-time.</p></li>
    <li><strong>Document Summarization:</strong> <p style="font-size:16px;">Generate concise summaries of uploaded research papers to quickly grasp the content.</p></li>
    <li><strong>Keyword Extraction:</strong> <p style="font-size:16px;">Use NLP techniques to extract important keywords from the document for quick topic identification.</p></li>
    <li><strong>Advanced Search:</strong> <p style="font-size:16px;">Search through the document using custom queries to find specific information or snippets of interest.</p></li>
    <li><strong>Memory-Enabled Conversations:</strong> <p style="font-size:16px;">Store and maintain chat history with the document using conversational memory to track your questions and responses.</p></li>
    <li><strong>PDF Preview:</strong> <p style="font-size:16px;">View and navigate through the uploaded PDF document, including the ability to download it.</p></li>
</ul>

<h2>How It Works:</h2>
<ol>
    <li><p style="font-size:16px;">Upload a PDF document.</p></li>
    <li><p style="font-size:16px;">The system parses the document, chunks it, and stores it in a vector database.</p></li>
    <li><p style="font-size:16px;">Interact with the chatbot by asking questions about the document.</p></li>
    <li><p style="font-size:16px;">The chatbot retrieves relevant document chunks and provides responses based on the content.</p></li>
</ol>

<h2>Tech Stack:</h2>
<ul>
    <li><p style="font-size:16px;">LangChain for chain building and document retrieval.</p></li>
    <li><p style="font-size:16px;">FAISS for vector similarity search.</p></li>
    <li><p style="font-size:16px;">Hugging Face Embeddings for text vectorization.</p></li>
    <li><p style="font-size:16px;">LLama 3.1 for language understanding and response generation.</p></li>
    <li><p style="font-size:16px;">Streamlit for building the web-based interface.</p></li>
    <li><p style="font-size:16px;">PyTorch for machine learning and backend support.</p></li>
</ul>

<h2>Installation:</h2>
<ol>
    <li><p style="font-size:16px;">Clone the repository:</p>
        <pre><code>git clone https://github.com/your-username/chat-with-research-paper.git</code></pre>
    </li>
    <li><p style="font-size:16px;">Install the required dependencies:</p>
        <pre><code>pip install -r requirements.txt</code></pre>
    </li>
    <li><p style="font-size:16px;">Run the Streamlit app:</p>
        <pre><code>streamlit run app.py</code></pre>
    </li>
</ol>
