<h1>🦙 Chat With Document 📄</h1>

<p style="font-size:18px;">This project focuses on building a chatbot with document retrieval capabilities powered by LangChain and LLama 3.1.</p>

<h2>Key Features:</h2>
<ul>
    <li><strong>PDF Upload and Parsing:</strong> <p style="font-size:16px;">Easily upload PDF files, extract, and analyze the content for conversational interactions.</p></li>
    <li><strong>Chunking and Vector Storage:</strong> <p style="font-size:16px;">Uses Hugging Face Embeddings to chunk text and store it in FAISS for efficient retrieval.</p></li>
    <li><strong>Conversational Chatbot:</strong> <p style="font-size:16px;">Powered by LLama 3.1, the chatbot answers questions based on the document content.</p></li>
    <li><strong>GPU/CPU Support:</strong> <p style="font-size:16px;">Detects GPU for faster processing or defaults to CPU for seamless performance.</p></li>
    <li><strong>Memory Integration:</strong> <p style="font-size:16px;">Conversation history is maintained using ConversationBufferMemory for continuous interactions.</p></li>
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
        <pre><code>git clone https://github.com/chat_with_doc</code></pre>
    </li>
    <li><p style="font-size:16px;">Install the required dependencies:</p>
        <pre><code>pip install -r requirements.txt</code></pre>
    </li>
    <li><p style="font-size:16px;">Run the Streamlit app:</p>
        <pre><code>streamlit run app.py</code></pre>
    </li>
</ol>
