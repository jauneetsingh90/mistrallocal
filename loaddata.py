from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import AstraDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains import RetrievalQA
from langchain.memory import ConversationSummaryMemory
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
import os
import streamlit as st


ASTRADB_TOKEN= os.getenv("ASTRADB_TOKEN")
ASTRADB_ENDPOINT= os.getenv("ASTRADB_ENDPOINT")



embeddings = HuggingFaceEmbeddings()


vectorstore=AstraDB(
        embedding=embeddings,
        collection_name="democol",
        token=ASTRADB_TOKEN,
        api_endpoint=ASTRADB_ENDPOINT
)


uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    temp_filepath= uploaded_file
    docs =[]
    loader = CSVLoader(file_path=temp_filepath)
    data = loader.load()
    vectorstore.add_documents(data)


st.success("Embeddings generated and added to the vector store.")

    
    
