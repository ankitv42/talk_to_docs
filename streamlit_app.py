import os
import tempfile
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain.document_loaders import PyPDFLoader
from langchain import hub

# Set API key (Replace with your actual key)
os.environ["GROQ_API_KEY"] = "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"

# Initialize LLM and Embeddings
llm = ChatGroq(model="llama3-8b-8192")
model_name = "BAAI/bge-small-en"
hf_embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Streamlit App
st.title("📄 PDF Chatbot with RAG")
st.write("Upload a PDF and ask questions!")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load and process PDF
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()

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

    st.success("PDF processed successfully! Now ask questions.")

    # Query input
    query = st.text_input("Ask a question about the PDF:")

    if st.button("Submit"):
        if query:
            response = rag_chain.invoke(query).content
            st.write("### AI Response:")
            st.write(response)
        else:
            st.warning("Please enter a question.")
