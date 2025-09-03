import sys
import os

sys.path.append(os.path.join(os.getcwd(),"..","Essence_Extractor"))
sys.path.append(os.path.join(os.getcwd(),"..","Entity_Extractor"))
sys.path.append(os.path.join(os.getcwd(),"..","Effect_Forecaster"))

import streamlit as st

from extract_document_essence import Extract_Document_Essence
from extract_entities import Extract_Entities
from effect_forecaster import Effect_Forecaster


st.header("Forecasting Effect")
st.write("Use AI to forecast the possible effects that certain statements in the document may have.")

st.write(" ")
st.write(" ")

c1,c2,c3=st.columns([1,2,1])
with c2:
    forecast_button=st.button("Forecast Effect")


st.write(" ")
st.write(" ")

if forecast_button:
    document=st.session_state.get("document")

    if document:
        if not st.session_state.get("forecasted_effects"):
            if not st.session_state.get("document_essence"):
                doc_essence=Extract_Document_Essence(document)
                st.session_state.document_essence=doc_essence

            else:
                doc_essence=st.session_state.get("document_essence")


            important_points=[]
            for point in doc_essence:
                point=point.page_content
                point=point.strip()
                point=point.capitalize()

                important_points.append(point)


            if not st.session_state.get("extracted_entities"):
                extracted_entities=Extract_Entities(document)
                st.session_state.extracted_entities=extracted_entities

            else:
                extracted_entities=st.session_state.get("extracted_entities")


            forecasted_effects=Effect_Forecaster(
                document_essence=important_points,
                entities_extracted=extracted_entities
            )

            st.session_state.forecasted_effects=forecasted_effects

            
        else:
            forecasted_effects=st.session_state.get("forecasted_effects")


        for inst in forecasted_effects:
            with st.expander(f"Statement: {inst['statement']}"):
                st.write(f"Effect: {inst['effect']}")

            st.write(" ")
            

    else:
         st.warning("Please upload a valid document")
        
            
            
                

