import sys
import os
sys.path.append(os.path.join(os.getcwd(),"..","Document_Classifier"))

import streamlit as st
import pandas as pd
import altair as alt

from classify_document import Classify_document

st.header("Document Classifier")

st.write("Using AI to classify the document into categories such as Politics, Business, Technology, Sports, and Entertainment")
st.write(" ")
st.write(" ")

c1,c2,c3=st.columns([1,2,1])

with c2:
    classify_button=st.button("Classify Document")

if classify_button:
    st.write(" ")
    st.write(" ")
    
    if st.session_state.get("document"):
        document=st.session_state.get("document")
        
        if not st.session_state.get("document_classification_results"):
            results=Classify_document(document)
            st.session_state.document_classification_results=results

        else:
            results=st.session_state.get("document_classification_results")
            

        df=pd.DataFrame(list(results.items()),columns=["Category","Probability"])
        df["Prob_percentage"]=df["Probability"]*100

        top_category=df.loc[df["Probability"].idxmax(),"Category"]

        chart=alt.Chart(df).mark_bar().encode(
            x=alt.X("Prob_percentage",title="Probability (%)"),
            y=alt.Y("Category",title="Category",sort="-x"),
            color=alt.condition(
                alt.datum.Category==top_category,
                alt.value("orange"),
                alt.value("steelblue")
            ),
            
        tooltip=[
            alt.Tooltip("Category",title="Category"),
            alt.Tooltip("Prob_percentage",title="Probability (%)")
        ]
        
        ).properties(
            width=600,
            height=300,
            title="Document Classification Results"
        )

        st.altair_chart(chart,use_container_width=True)

        top_percentage=df.loc[df["Probability"].idxmax(),"Prob_percentage"]

        st.write(" ")
        st.write(" ")

        st.success(f"Predicted Category: {top_category} ({top_percentage})")
        
        

    else:
        st.warning("Please upload a valid document")
