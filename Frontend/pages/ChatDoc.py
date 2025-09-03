import sys
import os

sys.path.append(os.path.join(os.getcwd(),"..","Essence_Extractor"))
sys.path.append(os.path.join(os.getcwd(),"..","ChatDoc"))

import streamlit as st

from embed_document import Embed_Document
from doc_chatbot import AskChatBot


st.header("ChatDoc")
st.write("Ask questions about your document")

st.write(" ")
st.write(" ")

c1,c2,c3=st.columns([1,2,1])
with c2:
    chat_button=st.button("Start Chat")

st.write(" ")
st.write(" ")


if "chat_messages" not in st.session_state:
    st.session_state.chat_messages=[]

if chat_button:
    document=st.session_state.get("document")
    
    if not document:
        st.warning("Please upload a valid document")

        
document=st.session_state.get("document")

if not st.session_state.get("vectorstore"):
    vectorstore=Embed_Document(document)
    st.session_state.vectorstore=vectorstore

else:
    vectorstore=st.session_state.get("vectorstore")

retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":3,
        "similarity_threshold":0.3,
        "lambda_mult":0.5
    }
)

    
user_input=st.chat_input("Ask something about document...")
if user_input:
    user_message={
        "role":"user",
        "content":user_input
    }
    
    context=retriever.invoke(user_input)
    context_points=[]
    for point in context:
        point=point.page_content
        point=point.strip()
        point=point.capitalize()

        context_points.append(point)

    

    ai_response=AskChatBot(
        question=user_input,
        context=context_points,
        chat_history=st.session_state.chat_messages
    )
    
    ai_message={
        "role":"assistant",
        "content":ai_response
    }
    
    st.session_state.chat_messages.append(user_message)
    st.session_state.chat_messages.append(ai_message)

st.write(" ")

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
    
    

        
