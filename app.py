import streamlit as st
import os
import shutil
import speech_recognition as sr
from rag_pipeline import RAGPipeline

st.set_page_config(page_title="FDA Cosmetic Guidance Chatbot", layout="wide")
st.title("FDA Cosmetic Guidance Chatbot")

st.sidebar.header("Configuration")
uploaded_file = st.sidebar.file_uploader("Upload the FDA Cosmetic Guidance PDF", type="pdf")
persona = st.sidebar.selectbox("Select Your Persona", ["Professional", "High School Student", "Layman", "Government Official"])
model_name = st.sidebar.selectbox("Select LLM Model", ["qwen2.5:0.5b-instruct-q4_0", "llama3.2:1b-instruct-q4_K_M"])
num_references = st.sidebar.slider("Number of References", min_value=1, max_value=5, value=3)

def record_audio(status_placeholder):
    recognizer = sr.Recognizer()
    recognizer.pause_threshold = 3.0  # Stop after 3 seconds of silence
    with sr.Microphone() as source:
        status_placeholder.write("Recording started... Speak your question now! (Stops after 3-second pause)")
        audio = recognizer.listen(source, timeout=None)
        status_placeholder.write("Processing audio...")
        try:
            text = recognizer.recognize_google(audio)
            status_placeholder.empty()  # Clear status after success
            return text
        except sr.UnknownValueError:
            status_placeholder.error("Sorry, I couldn't understand the audio.")
            return None
        except sr.RequestError as e:
            status_placeholder.error(f"Speech recognition service error: {str(e)}")
            return None

if uploaded_file:
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    file_path = os.path.join(data_dir, "uploaded_guidance.pdf")
    with open(file_path, "wb") as f:
        shutil.copyfileobj(uploaded_file, f)

    with st.spinner("Tokenizing and indexing the document..."):
        rag = RAGPipeline(file_path, model_name=model_name)
        rag.load_or_build_vector_store()
    st.sidebar.success("File uploaded successfully!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat history container
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Chat input container at the bottom
    input_container = st.container()
    with input_container:
        col1, col2 = st.columns([5, 1])
        with col1:
            prompt = st.chat_input("Ask a question about the FDA Cosmetic Guidance:")
        with col2:
            record_clicked = st.button("🎤", key="record_button")

        # Audio status placeholder below chatbar
        status_placeholder = st.empty()
        if record_clicked:
            prompt = record_audio(status_placeholder)

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        with st.spinner(f"Processing your question with {model_name}..."):
            answer, sources = rag.query(prompt, persona, num_references=num_references)
            response = f"**Answer**: {answer}\n\n**Sources**:\n" + "\n".join(sources)

        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.warning("Please upload the FDA Cosmetic Guidance PDF to start.")