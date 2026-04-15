import streamlit as st
import validators
from langchain_classic.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

# Streamlit APP

st.set_page_config(page_title="Langchain: Summarize Text from YT or Website")
st.title("Langchain : Summarize text from YT or Website")
st.subheader('Summarize URL')

# Get Groq API key and url field
with st.sidebar:
    groq_api_key = st.text_input("Gro API key", value ="", type="password")

# Initialize the llm
llm = ChatGroq(groq_api_key = groq_api_key,model = "llama-3.1-8b-instant")

# Prompt Template
prompt_tempalate = """
    Provide a summary of the following content in 250 words:
    Content:{text}

"""

prompt = PromptTemplate(template=prompt_tempalate,input_variables=['text'])


url = st.text_input("URL",label_visibility="collapsed")

if st.button("Summarize the content from YT or Website"):
    # Validate all the inputs
    if not groq_api_key.strip() or not url.strip():
        st.error("Plese provide the information")

    elif not validators.url(url):
        st.error("Please enter a valid url")

    else:
        try:
            with st.spinner("Waiting..."):
                # Loading the website data
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(url,add_video_info = True)
                else:
                    loader = UnstructuredURLLoader(urls = [url], ssl_verify = False,
                                                   headers={})
                    
                docs = loader.load()

                ## Chain for summarization
                chain = load_summarize_chain(llm, chain_type="stuff",prompt=prompt)
                summary = chain.run(docs)

                st.success(summary)
                
        except Exception as e:
            st.exception(f"Exception:{e}")