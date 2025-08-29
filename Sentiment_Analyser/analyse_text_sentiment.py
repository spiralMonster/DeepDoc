import sys
import os
sys.path.append(os.path.join(os.getcwd(),"..","Document_Classifier","Model_Training"))
sys.path.append(os.path.join(os.getcwd(),"..","Document_Classifier","Data_Preparations"))

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import CustomObjectScope

from clean_text_data import CleanText
from multi_lstm_unit import Multi_LSTM_Unit

data_store=Path(os.getcwd()).parent/"Data_Store"



def Analyse_Text_Sentiment(text):
    print(f"[Text Sentiment Analyser Pipeline started...]")
    
    cleaned_text=CleanText(text)
    print(f"[INFO] Text Cleaned")

    with open(os.path.join(data_store,"sentiment_analyser_tokenizer.json"),"r") as file:
        tokenizer_json=json.load(file)

    tokenizer=tokenizer_from_json(tokenizer_json)
    print(f"[INFO] Tokenizer Loaded")

    tokenized_text=tokenizer.texts_to_sequences([cleaned_text])
    print(f"[INFO] Text Tokenized")

    padded_text=pad_sequences(tokenized_text,maxlen=50,padding="post",truncating="post")
    print(f"[INFO] Text Padded")

    model_inp=np.reshape(padded_text,(1,5,10))
    model_inp=[model_inp[:,ind,:] for ind in range(5)]
    print(f"[INFO] Input for Model Prepared")

    with CustomObjectScope(
        {
            "Multi_LSTM_Unit":Multi_LSTM_Unit
        }
    ):
        model=load_model(
            os.path.join(
                data_store,
                "sentiment_analyser_model.h5"     
            )
        )

    print(f"[INFO] Sentiment Analyser Model loaded")

    predictions=model.predict(model_inp,batch_size=1)[0].tolist()
    print(f"[INFO] Predictions made")

    with open(os.path.join(data_store,"sentiment_analyser_label_mapping.json"),"r") as file:
        label_mapping=json.load(file)

    print(f"[INFO] Label Mappings Loaded")

    text_sentiment={}
    for label,preds in zip(label_mapping.keys(),predictions):
        text_sentiment[label]=preds

    print(f"[INFO] Text Sentiment: ")
    print(text_sentiment)

    print(f"[Text Sentiment Analyser Pipeline ended...]")

    return text_sentiment


if __name__=="__main__":
    df=pd.read_csv(os.path.join(data_store,"sentiment_analysis_dataset.csv"))
    text=df.iloc[2].Description

    print(f"The Text is: ")
    print(text)

    Analyse_Text_Sentiment(text)
