import sys
import os
sys.path.append(os.path.join(os.getcwd(),"..","Document_Classifier"))
sys.path.append(os.path.join(os.getcwd(),"..","Essence_Extractor"))
sys.path.append(os.path.join(os.getcwd(),"..","Entity_Extractor"))
sys.path.append(os.path.join(os.getcwd(),"..","Document_Insighter"))

import numpy as np
import streamlit as st

from classify_document import Classify_document
from extract_document_essence import Extract_Document_Essence
from extract_entities import Extract_Entities
from get_insights_from_document import GetDocumentInsights

st.header("Document Insighter")
st.write("Get to know what the document is about")

st.write("")
st.write("")

c1,c2,c3=st.columns([1,2,1])
with c2:
    insight_button=st.button("Get Insights")


st.write("")
st.write("")

if insight_button:
    document=st.session_state.get("document")
    
    if document:
        if not st.session_state.get("document_classification_results"):
            doc_results=Classify_document(document)
            st.session_state.document_classification_results=doc_results

        else:
            doc_results=st.session_state.get("document_classification_results")


        if not st.session_state.get("extracted_entities"):
            extracted_entities=Extract_Entities(document)
            st.session_state.extracted_entities=extracted_entities

        else:
            extracted_entities=st.session_state.get("extracted_entities")

        if not st.session_state.get("document_essence"):
            doc_essence=Extract_Document_Essence(document)
            st.session_state.document_essence=doc_essence

        else:
            doc_essence=st.session_state.get("document_essence")
            
            

        ind=int(np.argmax(np.asarray(doc_results.values()),axis=0))
        doc_type=list(doc_results.keys())[ind]

        important_points=[]
        for ind,point in enumerate(doc_essence):
            point=point.page_content
            point=point.strip()
            point=point.capitalize()

            important_points.append(point)

        if not st.session_state.get("document_insights"):
            insights=GetDocumentInsights(
                doc_type=doc_type,
                doc_essence=important_points,
                persons_mentioned=extracted_entities["persons"],
                organization_mentioned=extracted_entities["organization"],
                places_mentioned=extracted_entities["places"]
            )
            
            st.session_state.document_insights=insights
            
        else:
            insights=st.session_state.get("document_insights")
            
            

        for insight in insights:
            with st.expander(f"{insight['question']}"):
                st.write(insight['answer'])

    else:
         st.warning("Please upload a valid document")
            
            
        