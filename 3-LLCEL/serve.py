from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes # helps to create api's
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",groq_api_key=groq_api_key)

system_template = "Translate the following into {language}"
prompt_template = ChatPromptTemplate.from_messages([
    ('system',system_template),
    ('user','{text}')
])

parser = StrOutputParser()

# create chain
chain = prompt_template | model | parser

# App Definition
app = FastAPI(title="Langchain Server",
              version="1.0",
              description="A simple API server using Lanchain runnable interfaces")

# Adding chain routes
add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost",port=8080)
 
'''
 To run python serve.py
 and then go to url and add /docs at end in url
 similarly /chain/input_schema and others are there
 
 If I want I can use Postman to check the api or without it also
 '''