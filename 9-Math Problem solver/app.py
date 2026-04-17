import streamlit as st
from langchain_groq import ChatGroq

from langchain_classic.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents.agent_types import AgentType 
from langchain_classic.agents import initialize_agent 
from langchain_classic.tools import Tool
from dotenv import load_dotenv 
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_core.output_parsers import StrOutputParser

# Streamlit app
st.set_page_config(page_title="Text to math problem solver and data search assistant")
st.title("Text to Math Problem Solver")

groq_api_key= st.sidebar.text_input(label="GROQ API KEY",type="password")

if not groq_api_key:
    st.info("Please add your groq api key to continue")
    st.stop()

llm = ChatGroq(groq_api_key = groq_api_key, model="llama-3.3-70b-versatile")

# Initiliasing the tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name = "Wikipedia",
    func = wikipedia_wrapper.run,
    description = "A tool to search internet to find various information on the topic"
)

# Initialise the math tool
math_chain = llm | StrOutputParser()
calculator = Tool(
    name = "calculator",
    func = lambda x:math_chain.invoke(x),
    description = "A tool for math related questions. Only input mathematical expression need to be provided"
)

prompt = """
"You are an agent used for solving user mathematical question" 
"Logically arrive to a solution and display it pointwise"
"""

prompt_template = PromptTemplate(
    input_variables=['question'],
    template=prompt

)

# Combine all tools into chain
llm_chain =  prompt_template | llm

reasoning_tool = Tool(
    name = "Reasoning tool",
    func = lambda x : llm_chain.invoke(x),
    description = "Tool for answering logic-based and reasoning question"
)

# Initiliase the agents
assistant_agents  = initialize_agent(
    tools = [wikipedia_tool,calculator,reasoning_tool],
    llm = llm,
    agent = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose = False,
    handle_parsing_error = True
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content" : "Hi I am a Math Chatbot who can try to answer math questions"}
    ]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Interaction

question = st.text_area("Enter your question")

if st.button("Find my answer"):
    if question:
        with st.spinner("Generate response"):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write()

            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = assistant_agents.invoke( question, callbacks=[st_cb])

            st.session_state.messages.append({"role":'assistant', 'content' : response})
            st.write('Response :')
            st.success(response)

    else:
        st.warning("Please enter the question")