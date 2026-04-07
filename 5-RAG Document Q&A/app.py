import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()

# Load all the enviroment variables
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-8b-instant")

prompt= ChatPromptTemplate.from_template(
    """
Answer the question based on provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Question:{input}
"""
)

def create_vector_embeddings():
    # session state helps to remember the vector stores, also used in chat history
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # keeping my embedding in session state also
        st.session_state.loader = PyPDFDirectoryLoader("document") # Data ingestion step
        st.session_state.docs = st.session_state.loader.load() # document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50]) # actually splitting all the documents
        # just so it dont take much time I just say that start from 50
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        # saving all this in session state

# Helper to format documents (Replacement for "Stuff" logic)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

user_prompt = st.text_input("Enter your query from the document")

if st.button("Document Embedding"):
    create_vector_embeddings()
    st.write("Vector Database is ready")

if user_prompt:
    retriever = st.session_state.vectors.as_retriever()

    # Retrieve the documents
    docs = retriever.invoke(user_prompt)    

    rag_chain = prompt | llm | StrOutputParser()

    response = rag_chain.invoke({
        'context': format_docs(docs),
        'input': user_prompt
    })

    st.write(response)

    with st.expander("Document similarity search"):
        for doc in docs:
            st.write(doc.page_content)
            st.write('---------------------------------')