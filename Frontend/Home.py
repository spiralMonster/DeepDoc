from dotenv import load_dotenv
load_dotenv()

import streamlit as st

st.header("DeepDoc")
st.write("An AI powered document analysis")

st.write("")
st.write("")

if "document" not in st.session_state:
    st.session_state.document=None

if "document_classification_results" not in st.session_state:
    st.session_state.document_classification_results=None

if "extracted_entities" not in st.session_state:
    st.session_state.extracted_entities=None

if "document_essence" not in st.session_state:
    st.session_state.document_essence=None

if "text_sentiment" not in st.session_state:
    st.session_state.text_sentiment=None

if "document_insights" not in st.session_state:
    st.session_state.document_insights=None

if "forecasted_effects" not in st.session_state:
    st.session_state.forecasted_effects=None
    

if "extractor_triggered" not in st.session_state:
    st.session_state.extractor_triggered = False

if "doc_summary" not in st.session_state:
    st.session_state.doc_summary=None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore=None
    

uploaded_file = st.file_uploader("Upload your document", type=["txt"])


if uploaded_file is not None:
    st.success(f"File uploaded: {uploaded_file.name}")

    document=uploaded_file.read().decode("utf-8")
    st.session_state.document=document

    st.write("")
    st.write("")

    c1,c2,c3=st.columns([1,2,1])

    with c2:
        view_button=st.button("View Uploaded Document")

    if view_button:
        st.write(" ")
        st.write(" ")

        st.text_area("Document Preview",document,height=1000)
        # st.write(document)

        hide_button=st.button("Hide Uploaded Document")

        if hide_button:
            st.write(" ")


if not st.session_state.get("document"):
    st.session_state.document_classification_results=None
    st.session_state.extracted_entities=None
    st.session_state.document_essence=None
    st.session_state.text_sentiment=None
    st.session_state.document_insights=None
    st.session_state.forecasted_effects=None
    st.session_state.doc_summary=None
    st.session_state.vectorstore=None
    
        
