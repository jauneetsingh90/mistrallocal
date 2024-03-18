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

ASTRADB_TOKEN='AstraCS:XhhSeapeErChCSEkWbgeRhpU:3466c8c2c64d25a42ae65a8c7dd9647565ae2aa2357970226a6db2d49d6c93cb'
ASTRADB_ENDPOINT='https://f89515af-5e74-45d9-902f-51a53f75a4fb-ap-south-1.apps.astra.datastax.com'

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

    
    