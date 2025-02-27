# RAG Chatbot for Document-Based Q&A

## Project Description

The RAG Chatbot is a Python-based conversational AI tool designed to answer questions based on content from user-uploaded PDF documents, such as the FDA Cosmetic Guidance. Leveraging Retrieval-Augmented Generation (RAG), it combines document retrieval with generative AI to provide contextually grounded responses. The app supports text and audio input (via speech-to-text), multiple user personas (e.g., Professional, Layman), and customizable reference counts, making it versatile for various document types and user needs.

## Approach

The chatbot uses a RAG framework:
- **Document Processing**: PDFs are parsed with `PyPDFLoader`, split into 500-character chunks (100-character overlap) using `RecursiveCharacterTextSplitter`, and embedded with `all-MiniLM-L6-v2` into a FAISS vector store for efficient retrieval.
- **Retrieval**: FAISS retrieves the top `k` (default 3) relevant chunks based on cosine similarity to the user’s query.
- **Generation**: Ollama-hosted LLMs (`qwen2.5:0.5b-instruct-q4_0` or `llama3.2:1b-instruct-q4_K_M`) generate responses using retrieved context and persona-specific prompts.
- **Relevance**: No strict pre-filtering; post-response validation checks for ignorance and coherence, ensuring document-related questions are answered while tolerating speculative responses for unrelated ones.
- **UI**: Streamlit provides an interactive interface with chat history above a text/audio input bar, displaying single-line, numbered references.

## Instructions to Run the App

### Prerequisites
- **Python**: 3.11 or higher
- **Ollama**: Installed and running locally (see [Ollama GitHub](https://github.com/ollama/ollama))
- **Microphone**: For audio input (optional)
- **Conda**: Installed for environment management

### Setup

#### Clone the Repository
```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

### Create a Conda Environment
- Create and activate the environment:
  ```bash
  conda create -n rag_chatbot python=3.11
  conda activate rag_chatbot
  ```

- Install dependencies:
  ```pip install -r requirements.txt
  ```

### Set Up Environment Variables
- Create a .env file in the root directory:
  ```OLLAMA_HOST=http://localhost:11434
    CHUNK_SIZE=500
    CHUNK_OVERLAP=100
  ```

### Download LLM Models
- Ensure Ollama is installed, then pull models:
  ```ollama pull qwen2.5:0.5b-instruct-q4_0
    ollama pull llama3.2:1b-instruct-q4_K_M
  ```

### Running the App
- Start Ollama Server (in a separate terminal):
  ```ollama serve
  ```

- Launch the App:
  ```streamlit run app.py
  ```
 Open your browser at http://localhost:8501.
 Upload a PDF, select a persona and model, adjust the reference slider, and ask questions via text or audio (click "🎤").

### Dependencies
 See requirements.txt for a full list of Python packages.

### Notes
- Ensure sufficient memory (~6.9 GiB RAM recommended) for indexing large PDFs.
- Audio input requires a working microphone and internet for Google Speech API.