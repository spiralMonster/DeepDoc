import sys
import os
from pathlib import Path

sys.path.append(os.path.join(os.getcwd(),"..","Essence_Extractor"))
sys.path.append(os.path.join(os.getcwd(),"..","Sentiment_Analyser"))
sys.path.append(os.path.join(os.getcwd(),"..","Document_Classifier","Model_Training"))

import streamlit as st
import altair as alt
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

from multi_lstm_unit import Multi_LSTM_Unit
from extract_document_essence import Extract_Document_Essence
from analyse_text_sentiment import Analyse_Text_Sentiment

data_store=Path(os.getcwd()).parent/"Data_Store"


st.header("Essence Extractor")
st.write("Extract the important points from the document and analyse their sentiment using AI.")

st.write(" ")
st.write(" ")

c1,c2,c3=st.columns([1,2,1])
with c2:
    extractor_button=st.button("Extract Essence")


st.write(" ")
st.write(" ")

if extractor_button:
    st.session_state.extractor_triggered = True

if st.session_state.extractor_triggered:
    document=st.session_state.get("document")

    if document:
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
            

        if not st.session_state.get("text_sentiment"):
            
            with CustomObjectScope({
                "Multi_LSTM_Unit":Multi_LSTM_Unit
            }):
                model=load_model(
                    os.path.join(data_store,"sentiment_analyser_model.h5")
                )
                
    
            print(f"[INFO] Sentiment Analyser Model Loaded.")
    
            text_sentiments=[]
            for point in important_points:
                sentiment_result=Analyse_Text_Sentiment(text=point,model=model)
    
                text_sentiments.append(sentiment_result)

            st.session_state.text_sentiment=text_sentiments

        else:
            text_sentiments=st.session_state.get("text_sentiment")


        ind=1
        for imp_point,sentiment_result in zip(important_points,text_sentiments):
            
            with st.expander(f"{imp_point}"):
                c1,c2,c3=st.columns([1,2,1])
                with c2:
                    sentiment_button=st.button("Analyse Sentiment",key="id_"+str(ind))

                st.write(" ")
                if sentiment_button:
                    df=pd.DataFrame(list(sentiment_result.items()),columns=["Category","Probability"])
                    df["Probability_Percentage"]=df["Probability"]*100

                    top_category=df.loc[df["Probability"].idxmax(),"Category"]
                    top_category_prob=df.loc[df["Probability"].idxmax(),"Probability_Percentage"]

                    chart=alt.Chart(df).mark_bar().encode(
                        x=alt.X("Probability_Percentage",title="Probability (%)"),
                        y=alt.Y("Category",title="Category",sort="-x"),
                        color=alt.condition(
                            alt.datum.Category==top_category,
                            alt.value("orange"),
                            alt.value("steelblue")
                        ),
                        tooltip=[
                            alt.Tooltip("Category",title="Category"),
                            alt.Tooltip("Probability_Percentage",title="Probability (%)")
                        ]
                        
                    ).properties(
                        width=600,
                        height=300,
                        title="Sentiment Analysis"
                    )


                    st.altair_chart(chart,use_container_width=True)
                    st.write("")
                    

                    st.success(f"Predicted Category: {top_category} ({top_category_prob})")
    
                st.write(" ")
                st.write(" ")

                ind+=1

                
    else:
         st.warning("Please upload a valid document")
        
                
            
        
            
    