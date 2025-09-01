import os
from dotenv import load_dotenv

from langchain.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

from pydantic import BaseModel,Field
from typing_extensions import TypedDict,Annotated,List


class QuestionAnswerPair(TypedDict):
    question: Annotated[str,...,"The question asked"]
    answer: Annotated[str,...,"The answer to the question"]


class DocumentInsighterSpecs(BaseModel):
    doc_insights: List[QuestionAnswerPair]= Field(description="List of the question answer pairs")


def GetDocumentInsights(doc_type,doc_essence,persons_mentioned,organization_mentioned,places_mentioned):
    llm=ChatMistralAI(
        model="mistral-small-latest",
        temperature=0.2,
        max_retries=2,
        api_key=os.environ["MISTRAL_API_KEY"]
        
    ).with_structured_output(DocumentInsighterSpecs)

    template=""""
    You are provided with:
     - Type of Document
     - Important points in the Document
     - Persons,Organizations and Places mentioned in the document

    Your job is to generate the question-answer pair such that the person will get the insights about the document.

    Note:
     - Try to ask the general questions for which you can get the answers from the context.
     - Refrain from using the "Named Entities" in question generation.
     - Generate minimum three and maximum five question-answer pair.

    Document Type: {doc_type}
    
    Important Points:
    {important_points}

    Persons mentioned: {persons_mentioned}
    Organizations_mentioned: {organizations_mentioned}
    Places mentioned: {places_mentioned}
    """

    prompt=ChatPromptTemplate.from_template(template=template,
                              input_variable=["doc_type",
                                              "important_points",
                                              "persons_mentioned",
                                              "organizations_mentioned",
                                              "places_mentioned"])

    insighter_chain=prompt|llm
    results=insighter_chain.invoke({
        "doc_type":doc_type,
        "important_points":doc_essence,
        "persons_mentioned":persons_mentioned,
        "organizations_mentioned":organizations_mentioned,
        "places_mentioned":places_mentioned
    })

    doc_insights=results.doc_insights

    return results


if __name__=="__main__":
    import sys
    sys.path.append(os.path.join(os.getcwd(),"..","Document_Classifier"))
    sys.path.append(os.path.join(os.getcwd(),"..","Essence_Extractor"))
    sys.path.append(os.path.join(os.getcwd(),"..","Entity_Extractor"))
    
    from pathlib import Path
    import numpy as np
    import pandas as pd

    from classify_document import Classify_document
    from extract_document_essence import Extract_Document_Essence
    from extract_entities import Extract_Entities
    
    data_store=Path(os.getcwd()).parent/"Data_Store"
    df=pd.read_csv(
        os.path.join(data_store,"doc_classification_data.csv")
    )

    document=df.iloc[0].Text
    print("The document is: ")
    print(document)

    doc_classification=Classify_document(document)
    doc_ind=int(np.argmax(np.asarray(doc_classification.values()),axis=0))
    doc_type=list(doc_classification.keys())[doc_ind]
    print(f"Doc Type: {doc_type}")
    
    doc_essence=Extract_Document_Essence(document)
    important_points=[]
    print("Document Essence:")
    for ind,point in enumerate(doc_essence):
        point=point.page_content
        point=point.strip()
        point=point.capitalize()
        point=f"{ind+1}. "+point

        print(point)
        important_points.append(point)


    extracted_entities=Extract_Entities(document)
    persons_mentioned=extracted_entities['persons']
    organizations_mentioned=extracted_entities['organization']
    places_mentioned=extracted_entities['places']

    print(f"Persons mentioned: {persons_mentioned}")
    print(f"Organizations mentioned: {organizations_mentioned}")
    print(f"Places mentioned: {places_mentioned}")

    
    document_insights=GetDocumentInsights(
        doc_type=doc_type,
        doc_essence=important_points,
        persons_mentioned=persons_mentioned,
        organization_mentioned=organizations_mentioned,
        places_mentioned=places_mentioned
    )

    print("Document Insight: ")
    print(document_insights)
    # for insight in document_insights:
    #     print(f"Question: {insight['question']}")
    #     print(f"Answer: {insight['answer']}")
    #     print("\n")

    

    
    