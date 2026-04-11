import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community. vectorstores import FAISS 
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchResults # duckduck to search anything on internet
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_classic.agents import initialize_agent, AgentExecutor, AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
groq_api_key = os.getenv("GROQ_API_KEY")

# Arxiv and wikipedia tools
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchResults(name="Search")


# Streamlit app
st.title("Langchain - Chat with search")
'''
In this example, we're using StreamlitCallbackHandler| to display the thoughts and actions of the agent
 '''

# Sidebar
st.sidebar.title("Settings")

# I want chat history also
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role" : "assistant","content":"Hi, I am chatbot who can search web, how can I help you?"}
    ]  # save it in session state

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt) # whenever we are going to write a prompt it is going to be appended into this message

    llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
    streaming=True
    )
    tools = [search, arxiv,wiki]

    # I need to convert these tools into agents so I will be able to invoke it
    search_agent =initialize_agent(tools, llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)

    with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = search_agent.run(st.session_state.messages, callbacks=[st_cb])

            st.session_state.messages.append({'role': 'assistant',"content": response})
            st.write(response)

