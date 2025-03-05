import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv(find_dotenv())

# Constants
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
HF_TOKEN = os.environ.get("HF_TOKEN")

# Load PDF Files
def load_pdf_files(data_path):
    try:
        loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        st.write(f"✅ Debug: Loaded {len(documents)} PDF pages")
        return documents
    except Exception as e:
        st.error(f"⚠️ Error loading PDFs: {e}")
        return []

# Create Text Chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Load Embeddings Model
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store Embeddings in FAISS
def create_or_load_faiss_db():
    if os.path.exists(DB_FAISS_PATH):
        db = FAISS.load_local(DB_FAISS_PATH, get_embeddings_model(), allow_dangerous_deserialization=True)
    else:
        st.write("🚀 Debug: Creating new FAISS database...")
        documents = load_pdf_files(DATA_PATH)
        text_chunks = create_chunks(documents)
        db = FAISS.from_documents(text_chunks, get_embeddings_model())
        db.save_local(DB_FAISS_PATH)
        st.write("✅ Debug: FAISS database saved!")
    return db

# Load LLM Model (Mistral-7B) with Increased Output
def load_llm():
    try:
        llm = HuggingFaceEndpoint(
            repo_id=HUGGINGFACE_REPO_ID,
            temperature=0.5,  # More consistent responses
            model_kwargs={"token": HF_TOKEN, "max_length": 4096},  # Increased output length
        )
        
        return llm
    except Exception as e:
        st.error(f"⚠️ Error loading LLM: {e}")
        return None

# Custom Prompt Template for More Detailed Answers
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know—don't try to make up an answer. 
Be as detailed and informative as possible, explaining concepts step by step. 
Provide relevant examples, medical cases, or analogies when needed.

Context: {context}

Question: {question}

Please give a detailed, well-structured, and thorough response, divided into multiple paragraphs.
"""

def set_custom_prompt():
    return PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])

# Create Retrieval-Based QA Chain
def create_qa_chain():
    db = create_or_load_faiss_db()
    llm = load_llm()
    if llm is None:
        st.error("⚠️ LLM failed to load. Please check your API token and internet connection.")
        return None

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 6}),  # Increased number of retrieved chunks
        return_source_documents=True,
        chain_type_kwargs={"prompt": set_custom_prompt()},
    )
    return qa_chain

# Query Care Companion Model
def query_care_companion(user_query):
    qa_chain = create_qa_chain()
    if qa_chain is None:
        return "⚠️ Error: Unable to process query due to LLM failure.", []

    response = qa_chain.invoke({"query": user_query})
    return response["result"], response["source_documents"]

# Streamlit Web App
def main():
    st.title("CARE COMPANION 😪❤️")
    st.write("🩺 **Your AI-powered medical assistant!** Ask me anything related to medicine and health.")

    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Debugging checkpoint

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    prompt = st.chat_input("Pass your prompt here 👉🏻👉🏻:")

    if prompt:

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Query the LLM + FAISS Memory
        response, sources = query_care_companion(prompt)

        # Store assistant message
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

if __name__ == "__main__":
    main()
