import sys
import os
sys.path.append(os.path.join(os.getcwd(),".."))

from pathlib import Path
from dotenv import load_dotenv

import re
from langchain_chroma.vectorstores import Chroma
from langchain_text_splitters import TokenTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.vectorstores.in_memory import InMemoryVectorStore


load_dotenv()


def Embed_Document(doc):
    embedding_model=MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.environ["MISTRAL_API_KEY"]
    )

    
    vectorstore=InMemoryVectorStore(embedding_model)
    
    doc=re.sub(r"\n","",doc)
    doc=re.sub(r"\s\s+"," ",doc)

    splitter=TokenTextSplitter(
        chunk_size=50,
        chunk_overlap=10
    )

    splitted_texts=splitter.split_text(doc)

    vectorstore.add_texts(splitted_texts)

    return vectorstore

    

    
    