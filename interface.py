import sys, os
sys.dont_write_bytecode = True

import pandas as pd
import streamlit as st
import openai
from streamlit_modal import Modal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.embeddings import HuggingFaceEmbeddings

from llm_agent import ChatBot
from ingest_data import ingest
from retriever import SelfQueryRetriever
import chatbot_verbosity as chatbot_verbosity

import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))

DATA_PATH = "./data/main-data/synthetic-resumes.csv"
FAISS_PATH = "./vectorstore"
RAG_K_THRESHOLD = 5
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#LLM_MODEL = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
CUSTOMED_ENDPOINT = "http://localhost:1234/v1"
API_KEY = "lm-studio"



welcome_message = """"""

info_message = """"""

about_message = """"""


st.set_page_config(page_title="Resume Screening GPT")
st.title("Resume Screening GPT")

if "chat_history" not in st.session_state:
  st.session_state.chat_history = [AIMessage(content=welcome_message)]

if "df" not in st.session_state:
  st.session_state.df = pd.read_csv(DATA_PATH)

if "embedding_model" not in st.session_state:
  st.session_state.embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})

if "rag_pipeline" not in st.session_state:
  vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)
  st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)

if "resume_list" not in st.session_state:
  st.session_state.resume_list = []



def upload_file():
  modal = Modal(key="Demo Key", title="File Error", max_width=500)
  if st.session_state.uploaded_file != None:
    try:  
      df_load = pd.read_csv(st.session_state.uploaded_file)
    except Exception as error:
      with modal.container():
        st.markdown("The uploaded file returns the following error message. Please check your csv file again.")
        st.error(error)
    else:
      if "Resume" not in df_load.columns or "ID" not in df_load.columns:
        with modal.container():
          st.error("Please include the following columns in your data: \"Resume\", \"ID\".")
      else:
        with st.toast('Indexing the uploaded data. This may take a while...'):
          st.session_state.df = df_load
          vectordb = ingest(st.session_state.df, "Resume", st.session_state.embedding_model)
          st.session_state.retriever = SelfQueryRetriever(vectordb, st.session_state.df)
  else:
    st.session_state.df = pd.read_csv(DATA_PATH)
    vectordb = FAISS.load_local(FAISS_PATH, st.session_state.embedding_model, distance_strategy=DistanceStrategy.COSINE, allow_dangerous_deserialization=True)
    st.session_state.rag_pipeline = SelfQueryRetriever(vectordb, st.session_state.df)




def clear_message():
  st.session_state.resume_list = []
  st.session_state.chat_history = [AIMessage(content=welcome_message)]



user_query = st.chat_input("Type your message here...")

with st.sidebar:
  st.markdown("# Control Panel")

  # st.text_input("OpenAI's API Key", type="password", key="api_key")
  st.selectbox("RAG Mode", ["Generic RAG", "RAG Fusion"], placeholder="Generic RAG", key="rag_selection")
 
  st.file_uploader("Upload resumes", type=["csv"], key="uploaded_file", on_change=upload_file)
  st.button("Clear conversation", on_click=clear_message)

  # st.divider()
  # st.markdown(info_message)

  st.divider()
  # st.markdown(about_message)
  st.markdown("Made by Dharssini and Prasannapathi")


for message in st.session_state.chat_history:
  if isinstance(message, AIMessage):
    with st.chat_message("AI"):
      st.write(message.content)
  elif isinstance(message, HumanMessage):
    with st.chat_message("Human"):
      st.write(message.content)
  else:
    with st.chat_message("AI"):
      message[0].render(*message[1:])



retriever = st.session_state.rag_pipeline

llm = ChatBot(
  # api_key="lm-studio",
  # model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
)

if user_query is not None and user_query != "":
  with st.chat_message("Human"):
    st.markdown(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))

  with st.chat_message("AI"):
    start = time.time()
    with st.spinner("Generating answers..."):
      document_list = retriever.retrieve_docs(user_query, llm, st.session_state.rag_selection)
      query_type = retriever.meta_data["query_type"]
      st.session_state.resume_list = document_list
      stream_message = llm.generate_message_stream(user_query, document_list, [], query_type)
    end = time.time()

    response = st.write_stream(stream_message)
    
    retriever_message = chatbot_verbosity
    retriever_message.render(document_list, retriever.meta_data, end-start)

    st.session_state.chat_history.append(AIMessage(content=response))
    st.session_state.chat_history.append((retriever_message, document_list, retriever.meta_data, end-start))