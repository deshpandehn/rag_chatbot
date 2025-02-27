import os
import pickle
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from dotenv import load_dotenv
import re

load_dotenv()

class RAGPipeline:
    def __init__(self, file_path, model_name="llama3.2:1b-instruct-q4_K_M"):
        self.file_path = file_path
        self.llm = Ollama(model=model_name, base_url=os.getenv("OLLAMA_HOST"))
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vector_store = None
        self.cache_file = "faiss_cache.pkl"

    def load_and_split(self):
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", 500)),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", 100))
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    def load_or_build_vector_store(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                self.vector_store = pickle.load(f)
        else:
            chunks = self.load_and_split()
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.vector_store, f)

    def _extract_relevant_sentences(self, text, question, max_sentences=2):
        text = re.sub(r'\n|\r', ' ', text.strip())
        sentences = re.split(r'(?<=[.!?])\s+', text)
        question_words = set(question.lower().split())
        scored_sentences = []
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            relevance = len(question_words.intersection(sentence_words))
            if relevance > 0:
                scored_sentences.append((relevance, sentence.strip()))
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        relevant_sentences = [s[1] for s in scored_sentences[:max_sentences]]
        return " ".join(relevant_sentences) if relevant_sentences else sentences[0].strip()

    def query(self, question, persona, num_references=3):
        retriever = self.vector_store.as_retriever(search_kwargs={"k": num_references})
        docs = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in docs])
        persona_prompts = {
            "High School Student": "Explain in simple, beginner-friendly terms like you're teaching a high school student.",
            "Layman": "Provide a clear, everyday explanation suitable for someone with no technical background.",
            "Professional": "Give a detailed, technical response appropriate for an industry professional.",
            "Government Official": "Focus on regulatory compliance and policy implications, concise and formal."
        }
        prompt = f"{persona_prompts[persona]}\n\nBased only on this content from the FDA Cosmetic Guidance document:\n\n{context}\n\nAnswer this question: {question}"
        response = self.llm(prompt)
        sources = [f"{i+1}. Page {doc.metadata.get('page', 'Unknown')}: {self._extract_relevant_sentences(doc.page_content, question)}" for i, doc in enumerate(docs)]
        return response, sources