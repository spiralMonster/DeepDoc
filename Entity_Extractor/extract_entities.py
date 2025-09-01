import os
from pathlib import Path
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from typing_extensions import Annotated,TypedDict,List

load_dotenv()


class Entity_Extractor_Specs(TypedDict):
    persons: Annotated[List,...,"The persons mentioned in the text."]
    organization: Annotated[List,...,"The organizations mentioned in the text."]
    places: Annotated[List,...,"The places mentioned in the text."]


def Extract_Entities(document):
    llm=ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
        max_retries=2,
        api_key=os.environ["MISTRAL_API_KEY"]
        
    ).with_structured_output(Entity_Extractor_Specs)

    template="""
    You are given a document.
    Your job is to extract the persons,organizations and places mentioned in the text.
    
    Note:
     - If None are present then return an empty list.

    Document:
    {document}
    """

    prompt=ChatPromptTemplate.from_template(template=template,
                                            input_variable=["document"])

    entity_extractor_chain=prompt|llm

    results=entity_extractor_chain.invoke({
        "document":document
    })

    return results


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

    extracted_entities=Extract_Entities(document)
    print("The extracted entities are:")
    print(extracted_entities)
    