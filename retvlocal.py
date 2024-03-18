
import os
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import AstraDB
from langchain.chains import RetrievalQA
from langchain.memory import ConversationSummaryMemory
from langchain_openai import Ollama
from langchain.prompts import PromptTemplate

# Initialize embeddings and AstraDB
embeddings = HuggingFaceEmbeddings()

ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")

vectorstore = AstraDB(
    embedding=embeddings,
    collection_name="democol",
    token=ASTRA_DB_APPLICATION_TOKEN,
    api_endpoint=ASTRA_DB_API_ENDPOINT
)

# Define prompt template
template = """<s>[INST] Given the context - {context} </s>[INST] [INST] Answer the following question - {question}[/INST]"""
pt = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

# Initialize RetrievalQA model
rag = RetrievalQA.from_chain_type(
    llm=Ollama(model="mistral"),
    retriever=vectorstore.as_retriever(),
    memory=ConversationSummaryMemory(llm=Ollama(model="mistral")),
    chain_type_kwargs={"prompt": pt, "verbose": True},
)

# Streamlit UI
st.title("Chatbot Demo")

# Function to interact with the chatbot
def interact_with_chatbot():
    context = st.text_area("Context:", "")
    question = st.text_input("Question:", "")
    if st.button("Ask"):
        response = rag.ask(context=context, question=question)
        st.write("Chatbot:", response)

# Run the chatbot
interact_with_chatbot()

