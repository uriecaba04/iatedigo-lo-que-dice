import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import os


st.header("IA Te digo lo que dice tu documento")
st.write("¡Hola! Esta es una app con la que puedes interactuar con tu Documento")
OPENAI_API_KEY=st.text_input('Open AI API key', type="password")
pdf_obj=st.file_uploader("Carga tu documento", type="pdf")

def createEmbeding(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base

if(pdf_obj):
    knowledge_base = createEmbeding(pdf_obj)
    user_question=st.text_input("Ingresa tu pregunta")
    if(user_question):
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        docs = knowledge_base.similarity_search(user_question, 18)
        # Utilizar los parrafos similares para darle contexto a ChatGPT
        llm = ChatOpenAI(model_name='gpt-3.5-turbo')
        chain = load_qa_chain(llm, chain_type="stuff")
        respuesta = chain.run(input_documents=docs, question=user_question)
        st.write(respuesta)
        #st.write(docs)




hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: visible;}
            footer:after{
             content: 'Desarrollado Por Uriel Camargo';
             display:block;
             position:relative;
             color:tomato;
             
            }
            .appview-container{
              visibility: visible;
            }
            </style>
            """
hide_streamlit_style = """
            <style>
          
            footer {visibility: visible;}
            footer:after{
             content: 'Desarrollado Por Uriel Camargo';
             display:block;
             position:relative;
             color:tomato;
             
            }
            
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
