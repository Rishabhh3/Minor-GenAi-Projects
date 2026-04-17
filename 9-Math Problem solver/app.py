import streamlit as st
from langchain_groq import ChatGroq

from langchain_classic.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.agents.agent_types import AgentType 
from langchain_classic.agents import initialize_agent 
from langchain_classic.tools import Tool
from langchain_classic.callbacks import StreamlitCallbackHandler

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
# to wrap a function or callable, so can be used by agent as tool for specific tasks.
calculator = Tool(
    name = "calculator",
    func = lambda x:llm.invoke(x), # lambda so to make it runnable
    description = "A tool for math related questions. Only input mathematical expression need to be provided"
)
# the agent see the description and decides whether to use this tool or not

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
    func = llm_chain.invoke(x),
    description = "Tool for answering logic-based and reasoning question"
)

# Depending on use case I made 3 tools 
# calculator just feeds raw data to llm, like 2+2 no prompt required
# but reasoning tool adds prompt to it, now which tool to use
# for this we have description, I gave the list of tools to llm now based on
# description of tool , my llm(agent) will decide which tool to use and depending on
# the appropriate chain or tool will be used

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

# This loop displays the chat history in your Streamlit app.
# app shows a chat-like interface, displaying all previous messages in order, 
# with the correct sender (user or assistant) for each message.
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
            # receive update during execution, sends intm. outputs like show thinking
            # bridge between internal process and app's UI
            response = assistant_agents.invoke( question, callbacks=[st_cb])

            st.session_state.messages.append({"role":'assistant', 'content' : response})
            st.write('Response :')
            st.success(response)

    else:
        st.warning("Please enter the question")