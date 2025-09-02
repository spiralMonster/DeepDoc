import sys
import os

sys.path.append(os.path.join(os.getcwd(),"..","Entity_Extractor"))

import streamlit as st

from extract_entities import Extract_Entities

st.header("Entities Extractor")
st.write("Extract the persons,organizations and places mentioned in the document")

st.write(" ")
st.write(" ")

c1,c2,c3=st.columns([1,2,1])

with c2:
    extract_button=st.button("Extract Entities")

st.write(" ")
st.write(" ")

if extract_button:
    document=st.session_state.get("document")

    if document:
        if not st.session_state.get("extracted_entities"):
            extracted_entities=Extract_Entities(document)
            st.session_state.extracted_entities=extracted_entities

        else:
            extracted_entities=st.session_state.get("extracted_entities")
            

        persons_mentioned=",".join(extracted_entities["persons"])
        organizations_mentioned=",".join(extracted_entities["organization"])
        places_mentioned=",".join(extracted_entities["places"])

        with st.expander("Persons Mentioned:"):
            st.write(persons_mentioned)

        # st.write(" ")
        # st.write(" ")

        with st.expander("Organizations Mentioned: "):
            st.write(organizations_mentioned)

        # st.write(" ")
        # st.write(" ")

        with st.expander("Places Mentioned: "):
            st.write(places_mentioned)

        # st.write(" ")
        # st.write(" ")

    else:
        st.warning("Please upload a valid document")

        