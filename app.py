import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import os

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{question}")
])

@st.cache_resource
def load_llm(model, temperature, max_tokens):
    return Ollama(
        model=model,
        temperature=temperature,
        num_predict=max_tokens
    )

def generate_response(question, model, temperature, max_tokens):
    llm = load_llm(model, temperature, max_tokens)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})

st.title("Q&A Chatbot with Ollama")

model = st.sidebar.selectbox("Model", ["llama3", "llama2"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
max_tokens = st.sidebar.slider("Max Tokens", 50, 1000, 200, 50)

user_input = st.text_input("Ask something:")

if user_input:
    placeholder = st.empty()
    with st.spinner("🤖 Thinking..."):
        response = generate_response(user_input, model, temperature, max_tokens)
    placeholder.write(response)
else:
    st.write("Please enter a question to get started.")
