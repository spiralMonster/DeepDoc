import re
import nltk
import string
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

nltk.download("stopwords")
from nltk.corpus import stopwords

stop_words=set(stopwords.words("english"))



def CleanText(text,filter_stopwords=True):
    data=re.sub(r"\n","",text)
    data=re.sub(r"\s\s+"," ",data)

    final_text=""
    words=data.split(" ")

    table1=str.maketrans('','',string.punctuation)
    table2=str.maketrans('','',"0123456789")

    for word in words:
        if word.isalnum():
            word=word.lower()
            word=word.translate(table1)
            word=word.translate(table2)
            
            if filter_stopwords:
                if word not in stop_words:
                    final_text+=word
                    final_text+=" "
            else:
                final_text+=word
                final_text+=" "
                

    final_text=final_text.strip()
    return final_text
    