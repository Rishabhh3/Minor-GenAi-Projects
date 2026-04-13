import streamlit as st
import pathlib as Path
# in streamlit we have an agent that help us create sql agent itself
from langchain_classic.agents import create_sql_agent
from langchain_classic.sql_database import SQLDatabase
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_classic.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine 
import sqlite3
from langchain_groq import ChatGroq

st.set_page_config(page_title="Langchain : Chat with SQL DB")
st.title("Langchain: Chat with SQL DB")

# Problem with this is that there is problem of prompt injection
INJECTION_WARNING = """
            SQL agent can be vulnerable to prompt injection. Use a DB role with 
            limited permission.
"""

LOCAL_DB = "USE_LOCAL_DB"
MYSQL = "USE_MYSQL"

# radio options

radio_opt = ["Use SQLlite3 Database - Student.db","Connect your MySQL Database"]
selected_opt = st.sidebar.radio("Choose the db which you want to chat",options=radio_opt)

if radio_opt.index(selected_opt)==1: # second option
    db_url = MYSQL 
    mysql_host = st.sidebar.text_input("Provide MySQL host")
    mysql_user = st.sidebar.text_input("MySQL user")
    mysql_password = st.sidebar.text_input("MySQL Password", type = "password")
    mysql_db = st.sidebar.text_input("MySQL database")
else:
    db_url = LOCAL_DB

api_key = st.sidebar.text_input(label="GROQ API KEY", type="password")

if not db_url:
    st.info("Please enter the databse info and url")

if not api_key:
    st.info("Please enter GROQ API KEY")


# LLM MODEL
llm = ChatGroq(groq_api_key = api_key, model="llama-3.3-70b-versatile")

@st.cache_resource(ttl="1h")
def configure_db(db_url, mysql_host= None, mysql_user = None, mysql_password=None, mysql_db = None):
    if db_url == LOCAL_DB:
        # setup file path to local db
        db_file_path = (Path(__file__).parent/"student.db").absolute()
        print(db_file_path)

        creator = lambda: sqlite3.connect(f"file:{db_file_path}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///",creator=creator))
    
    elif db_url == MYSQL:
        if not(mysql_db and mysql_host and mysql_password and mysql_user):
            st.error("Please provide all MySQL connection details.")
            st.stop()

        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))

if db_url == MYSQL:
    db = configure_db(db_url,mysql_host,mysql_user,mysql_password,mysql_db)
else:
    db = configure_db(db_url)