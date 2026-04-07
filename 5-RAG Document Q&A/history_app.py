# RAG Q&A conversation history with PDF inlcuding chat history
# Optimization Possible - For every question It runs the entire script again even for same document so caching in streamlit can be used

import  streamlit as st
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

# set up streamlit app
st.title("Conversation RAG with pdf upload and history")
st.write("Upload PDF and chat with their content")

# Input with groq api key
groq_api_key = os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-8b-instant")

session_id = st.text_input("Session ID", value="default_session")

# statefully manage chat history

if 'store' not in st.session_state:
    st.session_state.store = {}

uploaded_file = st.file_uploader("Choose a pdf file", type="pdf", accept_multiple_files=True)

# Process pdf file
if uploaded_file:
    documents =[]
    for uploaded_file in uploaded_file:
    # we need to create temp storage to store pdf
        temppdf = f"./temp.pdf"
        with open(temppdf,"wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name

    loader = PyPDFLoader(temppdf)
    docs = loader.load()
    documents.extend(docs)


    # Split and create embeddings for the documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 5000, chunk_overlap = 500)
    splits = text_splitter.split_documents(documents)
    vector_store = Chroma.from_documents(documents = splits, embedding = HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2"))
    retriever = vector_store.as_retriever()

    # context prompt

    contextualize_q_system = (
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood."
        "without the chat history. Do NOT answer the question,"
        "just reformulate it if needed and otherwise return it as is"
    )

    contexualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system),
            MessagesPlaceholder("chat_history"),
            ("human","{input}"),
        ]
    )

    history_aware_retriever = contexualize_q_prompt | llm | StrOutputParser() | retriever

    # Answer question 
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "In\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}"),
        ]
    )
    question_answer_chain = (
        lambda x: {
        "context": "\n\n".join(doc.page_content for doc in x["context"]), 
        "input": x["input"],
        "chat_history": x["chat_history"]
        }
    ) | qa_prompt | llm | StrOutputParser()

    # RAG CHAIN
    rag_chain = RunnablePassthrough.assign(
        context = history_aware_retriever
    ) | question_answer_chain

    def get_session_history(session_id) ->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key=None
    )

    user_input = st.text_input("question")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": session_id}
            } # constructs a key "abc in store"
        )
        st.write(st.session_state.store)
        st.success(f"Assistant:: {response}")
        st.write("Chat History:", session_history.messages)
        st.write(f"Current Session ID: {session_id}")
        st.write(f"Sessions in Store: {list(st.session_state.store.keys())}")