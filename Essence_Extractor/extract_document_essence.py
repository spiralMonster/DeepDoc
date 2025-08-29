import os

from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings

from named_entity_extractor import Named_Entity_Extractor
from question_generator import Question_Generator
from context_creator import Context_Creator
from embed_document import Embed_Document

VECTOR_STORE=Path(os.getcwd()).parent/"Data_Store"/"vectorstore"

load_dotenv()



def Extract_Document_Essence(document,num_points=5):
    print(f"[Document Essence Extraction Pipeline Started...]")

    named_entities=Named_Entity_Extractor(document)
    print(f"[INFO] Named Entities Extracted from document")

    generated_questions=Question_Generator(named_entities)
    print(f"[INFO] Questions generated regarding named entities")

    context_doc=Context_Creator(generated_questions)
    print(f"[INFO] Context Document for Essence Extraction created")

    Embed_Document(document)
    print(f"[INFO] Document Embedded")

    embedding_model=MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.environ["MISTRAL_API_KEY"]
    )
    vectorstore=Chroma(
        embedding_function=embedding_model,
        persist_directory=VECTOR_STORE
    )
    print(f"[INFO] Vectorstore Loaded")

    retriever=vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":num_points,
            "fetch_k":num_points*2,
            "lambda_mult":0.5
        }
    )

    print(f"[INFO] Retrieving essence from the document using generated context document")
    extracted_essence=retriever.invoke(context_doc)

    print(f"[INFO] Essence extracted from Document.")
    
    print(f"[Document Essence Extraction Pipeline Completed...]")

    return extracted_essence


if __name__=="__main__":
    import pandas as pd
    data_store=Path(os.getcwd()).parent/"Data_Store"

    df=pd.read_csv(
        os.path.join(
            data_store,"doc_classification_data.csv"
        )
    )

    document=df.iloc[0].Text
    print("The document is: ")
    print(document)

    extracted_essence=Extract_Document_Essence(document)
    print("The Essence of the document is: ")
    for ind,point in enumerate(extracted_essence):
        print(f"{ind}. {point.page_content}")