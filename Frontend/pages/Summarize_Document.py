import sys
import os
sys.path.append(os.path.join(os.getcwd(),"..","Summarizer"))

import streamlit as st

from summarize_document import Summarize_Document

st.header("Document Summarizer")
st.write("Summarize the document")

st.write(" ")
st.write(" ")

c1,c2,c3=st.columns([1,2,1])
with c2:
    summarize_button=st.button("Get Summary")

st.write(" ")
st.write(" ")

if summarize_button:
    document=st.session_state.get("document")

    if document:
        if not st.session_state.get("doc_summary"):
            doc_summary=Summarize_Document(document)
            st.session_state.doc_summary=doc_summary

        else:
            doc_summary=st.session_state.get("doc_summary")

        st.text_area("Summary",doc_summary,height=200)

    else:
        st.warning("Please upload a valid document")
    