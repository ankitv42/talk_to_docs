import streamlit as st
import os
import tempfile
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import PyPDFLoader
from langchain import hub

# Set API key (replace with your actual key)
os.environ["GROQ_API_KEY"] = "your_groq_api_key"

# Streamlit UI
st.title("📄 PDF Chatbot with RAG")
st.write("Upload a PDF and ask questions!")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load and process PDF
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

    # Initialize LLM and Embeddings
    llm = ChatGroq(model="llama3-8b-8192")
    model_name = "BAAI/bge-small-en"
    hf_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Split text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create FAISS vector store
    vectorstore = FAISS.from_documents(documents=splits, embedding=hf_embeddings)
    retriever = vectorstore.as_retriever()

    # Load RAG prompt
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # RAG Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    # User Query
    user_query = st.text_input("Ask a question from the PDF:")

    if user_query:
        response = rag_chain.invoke(user_query)
        st.write("### 🤖 AI Response:")
        st.write(response)
