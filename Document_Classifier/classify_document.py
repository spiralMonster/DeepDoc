import sys
import os
import json
from pathlib import Path
sys.path.append(os.path.join(os.getcwd(),".."))

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.utils import CustomObjectScope
from Model_Training.multi_lstm_unit import Multi_LSTM_Unit
from Data_Preparations.clean_text_data import CleanText

data_store=Path(os.getcwd()).parent/"Data_Store"


def Classify_document(text):
    print(f"[Document Classification Pipeline Started...]")
    
    cleaned_text=CleanText(text)
    print(f"[INFO] Data Cleaned")

    with open(os.path.join(data_store,"doc_classifier_tokenizer.json"),"r") as file:
        tokenizer_json=json.load(file)

    tokenizer=tokenizer_from_json(tokenizer_json)
    print(f"[INFO] Tokenizer Loaded")

    tokenized_text=tokenizer.texts_to_sequences([cleaned_text])
    print(f"[INFO] Data Tokenized")


    pad_seq=pad_sequences(tokenized_text,maxlen=200,padding="post",truncating="post")
    print(f"[INFO] Data Padded")

    data=np.reshape(pad_seq,(1,8,25))
    inp_data=[data[:,ind,:] for ind in range(8)]
    print(f"[INFO] Data Prepared for Model Prediction")

    with CustomObjectScope(
        {
            "Multi_LSTM_Unit":Multi_LSTM_Unit
        }
    ):
        model=load_model(os.path.join(data_store,"document_classifier_model.h5"))

    print(f"[INFO] Model Loaded")

    predictions=model.predict(inp_data,batch_size=1)[0].tolist()

    with open(os.path.join(data_store,"doc_classifier_label.json"),"r") as file:
        labels=json.load(file)

    print(f"[INFO] Prediction Labels Loaded")

    document_classification={}
    for label,preds in zip(labels.keys(),predictions):
        document_classification[label]=preds

    print(f"[INFO] Document Classification:")
    print(document_classification)

    print(f"[Document Classification Pipeline Ended...]")

    return document_classification



if __name__=="__main__":
    data=pd.read_csv(
        os.path.join(data_store,"doc_classification_data.csv")
    )

    text=data.iloc[0].Text
    print(f"The Document is:")
    print(text)

    Classify_document(text)

    
    
